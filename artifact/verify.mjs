import fs from 'node:fs/promises';
import { createEncoder, createGPT } from './model.mjs';

const root = new URL('../', import.meta.url);
const [encoderData, gptData] = await Promise.all([
  fs.readFile(new URL('encoder_weights.json', root), 'utf8').then(JSON.parse),
  fs.readFile(new URL('gpt_weights.json', root), 'utf8').then(JSON.parse)
]);

const encoder = createEncoder(encoderData);
const gpt = createGPT(gptData);
const sentiment = encoder.analyze('this movie is surprisingly good');
const positive = encoder.analyze('a wonderful and touching film');
const negative = encoder.analyze('this movie is terrible');
const tokenized = gpt.tokenize('the film');
const causal = gpt.forwardAll(tokenized.idxs);
const generatedLow = gpt.generate('the film', 0.5, 6, () => 0.42);
const generatedHigh = gpt.generate('the film', 1.5, 6, () => 0.42);

const output = {
  encoderLogits: sentiment.logits,
  encoderAttention: sentiment.attention,
  encoderTokens: sentiment.tokens,
  gptLogits: causal.logits,
  gptAttention: causal.attention,
  generatedLow: generatedLow.text,
  generatedHigh: generatedHigh.text
};

if (process.argv.includes('--json')) {
  process.stdout.write(JSON.stringify(output));
} else {
  const upperMax = Math.max(...causal.attention.flatMap(map => map.flatMap((row, i) => row.slice(i + 1))));
  if (Object.keys(encoderData.weights).length !== 15 || Object.keys(gptData.weights).length !== 15) throw new Error('Cantidad de tensores exportados incorrecta.');
  if (sentiment.attention.length !== 2 || sentiment.attention[0].length !== sentiment.tokens.length) throw new Error('Forma de atención del encoder incorrecta.');
  if (causal.logits.length !== 14 || causal.logits[0].length !== 432) throw new Error('Forma de logits GPT incorrecta.');
  if (upperMax !== 0) throw new Error('La máscara causal permite atención futura.');
  if (positive.prediction !== 'POSITIVO' || negative.prediction !== 'NEGATIVO') throw new Error('Las predicciones de referencia cambiaron.');
  console.log('OK: 15 tensores por modelo cargados y dimensiones validadas.');
  console.log(`OK: encoder logits=[${sentiment.logits.map(value => value.toFixed(6)).join(', ')}], atención=2×${sentiment.tokens.length}×${sentiment.tokens.length}.`);
  console.log(`OK: referencias: POSITIVO=${(positive.confidence * 100).toFixed(1)}%, NEGATIVO=${(negative.confidence * 100).toFixed(1)}%.`);
  console.log(`OK: GPT logits=14×432, atención=2×14×14, máximo causal futuro=${upperMax}.`);
  console.log(`OK: muestreo determinista T=0.5: "${generatedLow.text}"`);
  console.log(`OK: muestreo determinista T=1.5: "${generatedHigh.text}"`);
}
