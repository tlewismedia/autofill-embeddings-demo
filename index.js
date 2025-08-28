import OpenAI from "openai";
import dotenv from "dotenv";
dotenv.config();

const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

async function batchEmbeddings() {
  const inputs = ["first name", "surname", "work email", "given name"];

  const response = await client.embeddings.create({
    model: "text-embedding-3-small",
    input: inputs,
  });

  // response.data is an array of objects with embeddings
  response.data.forEach((item, i) => {
    console.log(inputs[i], item.embedding.slice(0, 5)); // show first 5 numbers
  });
}

batchEmbeddings();
