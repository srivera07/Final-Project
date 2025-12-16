import OpenAI from "openai";
import dotenv from "dotenv";

dotenv.config();

const openai = new OpenAI();

async function run() {
  const image = await openai.images.generate({
    model: "gpt-image-1",
    prompt: "A futuristic cyberpunk city at night",
    size: "1024x1024",
  });

  console.log("IMAGE URL:");
  console.log(image.data[0].url);
}

run();
