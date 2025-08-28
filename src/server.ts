import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import OpenAI from "openai";
import type { Request, Response } from "express";
// Load environment variables
dotenv.config();

const app = express();
app.use(cors());
app.use(express.json());

// Define the structure of ThesaurusItem
interface ThesaurusItem {
  label: string;
  embedding: number[];
  canonical?: string;
}
const thesaurus: ThesaurusItem[] = [];

// Create OpenAI client
const client = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY || "",
});

// Check if label is not in thesaurus
function notInThesaurus(label: string): boolean {
  return !thesaurus.find((item) => item.label === label);
}

// Normalize strings
function normalizeString(str: string): string {
  return str
    .toLowerCase()
    .replace(/[^a-z]+/g, "_")
    .replace(/_+$/, "");
}

// Calculate cosine similarity
function cosineSimilarity(a: number[], b: number[]): number {
  const dot = a.reduce((sum, ai, i) => sum + ai * b[i], 0);
  const normA = Math.sqrt(a.reduce((sum, ai) => sum + ai * ai, 0));
  const normB = Math.sqrt(b.reduce((sum, bi) => sum + bi * bi, 0));
  return dot / (normA * normB);
}

// Find best match or use original
function findBestMatchOrUseOriginal(
  item: ThesaurusItem,
  threshold: number
): ThesaurusItem {
  const canonicalEmbeddings = thesaurus.map((t) => t.embedding);
  const scores = canonicalEmbeddings.map((embedding) =>
    cosineSimilarity(item.embedding, embedding)
  );

  const bestIndex = scores.indexOf(Math.max(...scores));
  if (bestIndex === -1 || scores[bestIndex] < threshold) {
    return item;
  }

  return thesaurus[bestIndex];
}

// Process labels
async function processLabels(
  labels: string[],
  threshold: number
): Promise<ThesaurusItem[]> {
  const newLabels = labels.filter(notInThesaurus);
  const response = await client.embeddings.create({
    model: "text-embedding-3-small",
    input: newLabels,
  });

  const embeddings = response.data.map((item, i) => ({
    label: newLabels[i],
    embedding: item.embedding,
  }));

  const normalizedEmbeddings = embeddings.map((item) => ({
    ...item,
    canonical: normalizeString(item.label),
  }));

  return normalizedEmbeddings.map((item) => {
    const similarOrOriginalItem = findBestMatchOrUseOriginal(item, threshold);
    const canonical = similarOrOriginalItem.canonical ?? item.canonical;
    return { ...item, canonical };
  });
}

// Prepare readable output
function makeReadable(item: ThesaurusItem): {
  label: string;
  canonical: string | undefined;
} {
  return { label: item.label, canonical: item.canonical };
}

// Handle the POST request to /thesaurus
app.post("/thesaurus", async (req: Request, res: Response) => {
  try {
    const { labels, threshold }: { labels: string[]; threshold: number } =
      req.body;
    const results = await processLabels(labels, threshold);

    thesaurus.push(...results);

    res.json(thesaurus.map(makeReadable));
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Failed to add to thesaurus" });
  }
});

// Start the server
const PORT = 3000;
app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});
