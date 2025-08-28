import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import OpenAI from "openai";

dotenv.config();

const app = express();
app.use(cors());
app.use(express.json());

const thesaurus = [];

// Create OpenAI client
const client = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

function notInThesaurus(label) {
  return !thesaurus.find((item) => item.label === label);
}

function normalizeString(str) {
  return str.toLowerCase().replace(/\s+/g, "_");
}

function cosineSimilarity(a, b) {
  const dot = a.reduce((sum, ai, i) => sum + ai * b[i], 0);
  const normA = Math.sqrt(a.reduce((sum, ai) => sum + ai * ai, 0));
  const normB = Math.sqrt(b.reduce((sum, bi) => sum + bi * bi, 0));
  return dot / (normA * normB);
}

// returns best match from thesaurus or original if not good enough
function findBestMatchOrUseOriginal(item) {
  console.log("in findBestMatch");

  const canonicalEmbeddings = thesaurus.map((item) => item.embedding);

  const scores = canonicalEmbeddings.map((embedding) =>
    cosineSimilarity(item.embedding, embedding)
  );

  console.log("scores");
  console.log(scores);

  const bestIndex = scores.indexOf(Math.max(...scores));

  if (bestIndex === -1) {
    console.log("thesaurus");
    return item;
  }

  const bestScore = scores[bestIndex];
  const bestMatchItem = thesaurus[bestIndex];

  if (bestScore < 0.5) {
    console.log("not good enough");
    return item;
  }

  return bestMatchItem;
}

// function findBestMatchOrUseOriginal(item) {
//   // todo: make pure function
//   if (thesaurus.length === 0) {
//     return false;
//   }

//   console.log("replaceWithMostSimilarIfFound");
//   console.log("label: ", item);

//   const bestMatch = findBestMatch(item);
//   console.log("bestMatch");
//   console.log(bestMatch);
//   return bestMatch;
// }

async function processLabels(labels) {
  //find any that aren’t in the thesaurus
  console.log("in processLabels");

  const newLabels = labels.filter(notInThesaurus);

  const response = await client.embeddings.create({
    model: "text-embedding-3-small", // cheaper + smaller
    input: newLabels,
  });

  console.log("response from processLabels");
  console.log(response);

  const embeddings = response.data.map((item, i) => {
    return {
      label: newLabels[i],
      embedding: item.embedding,
    };
  });

  console.log("embeddings from processLabels");
  console.log(embeddings);

  const normalizedEmbeddings = embeddings.map((item) => {
    return {
      ...item,
      canonical: normalizeString(item.label),
    };
  });

  console.log("normalizedEmbeddings from processLabels");
  console.log(normalizedEmbeddings);

  // for each embedding, check if there is a similar canonical, if so use it
  const newThesaurusItems = normalizedEmbeddings.map((item) => {
    const similarFoundOrOriginalItem = findBestMatchOrUseOriginal(item);
    const canonical = similarFoundOrOriginalItem.canonical;

    return {
      ...item,
      canonical,
    };
  });

  console.log("newThesaurusItems");
  console.log(newThesaurusItems);

  return newThesaurusItems;
  //get embeddings for each
  //normalize canonical
  //check for most similar canonical
  //use existing or new canonical depending on confidence value
}

// Route to get embedding
app.post("/embed", async (req, res) => {
  try {
    const { text } = req.body;

    const response = await client.embeddings.create({
      model: "text-embedding-3-small", // cheaper + smaller
      input: text,
    });

    res.json({
      text,
      embedding: response.data[0].embedding,
    });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Failed to get embedding" });
  }
});

function makeReadable(item) {
  return {
    label: item.label,
    canonical: item.canonical,
  };
}

// route to add to thesaurus
app.post("/thesaurus", async (req, res) => {
  //   console.log("thesaurus");
  //   console.log(req.body);
  try {
    const { labels } = req.body;

    const results = await processLabels(labels);

    thesaurus.push(...results);

    res.json({
      ...thesaurus.map(makeReadable),
    });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Failed to add to thesaurus" });
  }
});

const PORT = 3000;
app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});
