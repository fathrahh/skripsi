import { ModelFeatures } from "../components/FeaturesForm";

export default async function requestPrediction(
  features: ModelFeatures
): Promise<number> {
  const response = await fetch("http://127.0.0.1:8000/predict", {
    method: "POST",
    body: JSON.stringify(features),
    headers: {
      "content-type": "application/json",
    },
  });

  const data = (await response.json()) as number;

  return data;
}
