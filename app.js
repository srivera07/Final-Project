let model;

async function loadModel() {
  model = await mobilenet.load();
  console.log("MobileNet loaded");
}

document.getElementById("classifyBtn").addEventListener("click", async () => {
  const img = document.getElementById("preview");
  const predictionEl = document.getElementById("prediction");

  if (!img.src) {
    predictionEl.textContent = "Please upload an image first.";
    return;
  }

  const predictions = await model.classify(img);
  const top = predictions[0];

  predictionEl.textContent = `${top.className} (${(top.probability * 100).toFixed(2)}%)`;
});

document.getElementById("imageUpload").addEventListener("change", (e) => {
  const file = e.target.files[0];
  const img = document.getElementById("preview");
  img.src = URL.createObjectURL(file);
});

loadModel();
