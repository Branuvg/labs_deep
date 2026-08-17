const MAX_LEN_CLS = 16;
const MAX_LEN_GPT = 14;
const EPS = 1e-5;

function positionalEncoding(maxLen, dModel) {
  return Array.from({ length: maxLen }, (_, pos) =>
    Array.from({ length: dModel }, (_, i) => {
      const div = Math.exp(Math.floor(i / 2) * 2 * (-Math.log(10000) / dModel));
      return i % 2 === 0 ? Math.sin(pos * div) : Math.cos(pos * div);
    })
  );
}

function matmul(a, b) {
  const rows = a.length;
  const inner = b.length;
  const cols = b[0].length;
  const out = Array.from({ length: rows }, () => Array(cols).fill(0));
  for (let i = 0; i < rows; i++) {
    for (let k = 0; k < inner; k++) {
      const value = a[i][k];
      for (let j = 0; j < cols; j++) out[i][j] += value * b[k][j];
    }
  }
  return out;
}

function add(a, b) {
  return a.map((row, i) => row.map((value, j) => value + b[i][j]));
}

function addBias(a, bias) {
  return a.map(row => row.map((value, j) => value + bias[j]));
}

function layerNorm(x, gamma, beta) {
  return x.map(row => {
    const mean = row.reduce((sum, value) => sum + value, 0) / row.length;
    const variance = row.reduce((sum, value) => sum + (value - mean) ** 2, 0) / row.length;
    const scale = 1 / Math.sqrt(variance + EPS);
    return row.map((value, i) => gamma[i] * (value - mean) * scale + beta[i]);
  });
}

function softmax(values) {
  const max = Math.max(...values);
  const exp = values.map(value => value === -Infinity ? 0 : Math.exp(value - max));
  const total = exp.reduce((sum, value) => sum + value, 0);
  return exp.map(value => value / total);
}

function attention(x, weights, nHeads, dK, keyMask = null, causal = false) {
  const T = x.length;
  const Q = matmul(x, weights.WQ);
  const K = matmul(x, weights.WK);
  const V = matmul(x, weights.WV);
  const maps = [];
  const headOutputs = [];

  for (let h = 0; h < nHeads; h++) {
    const offset = h * dK;
    const scores = Array.from({ length: T }, (_, i) =>
      Array.from({ length: T }, (_, j) => {
        if ((keyMask && !keyMask[j]) || (causal && j > i)) return -Infinity;
        let score = 0;
        for (let k = 0; k < dK; k++) score += Q[i][offset + k] * K[j][offset + k];
        return score / Math.sqrt(dK);
      })
    );
    const map = scores.map(softmax);
    maps.push(map);
    headOutputs.push(Array.from({ length: T }, (_, i) =>
      Array.from({ length: dK }, (_, k) => {
        let value = 0;
        for (let j = 0; j < T; j++) value += map[i][j] * V[j][offset + k];
        return value;
      })
    ));
  }

  const concatenated = Array.from({ length: T }, (_, i) =>
    headOutputs.flatMap(head => head[i])
  );
  return { output: matmul(concatenated, weights.WO), attention: maps };
}

function transformerBlock(idxs, pe, weights, nHeads, dK, keyMask, causal) {
  const x = idxs.map((idx, i) => weights.E[idx].map((value, j) => value + pe[i][j]));
  const attended = attention(x, weights, nHeads, dK, keyMask, causal);
  const x2 = layerNorm(add(x, attended.output), weights.gamma1, weights.beta1);
  const hidden = addBias(matmul(x2, weights.W1), weights.bias1)
    .map(row => row.map(value => Math.max(0, value)));
  const ff = addBias(matmul(hidden, weights.W2), weights.bias2);
  const x3 = layerNorm(add(x2, ff), weights.gamma2, weights.beta2);
  return { encoded: x3, attention: attended.attention };
}

function validateExport(data, expectedType, outputWeight) {
  const required = ['E', 'WQ', 'WK', 'WV', 'WO', 'gamma1', 'beta1', 'W1', 'bias1', 'W2', 'bias2', 'gamma2', 'beta2', outputWeight];
  if (!data || data.model_type !== expectedType || !Array.isArray(data.vocab)) {
    throw new Error(`El archivo de pesos ${expectedType} no tiene el formato esperado.`);
  }
  for (const key of required) {
    if (!Array.isArray(data.weights?.[key])) throw new Error(`Falta el tensor ${key} en los pesos ${expectedType}.`);
  }
  if (data.d_model !== 32 || data.d_ff !== 64 || data.n_heads !== 2 || data.d_k !== 16) {
    throw new Error(`Las dimensiones exportadas de ${expectedType} no coinciden con el notebook.`);
  }
}

export function createEncoder(data) {
  validateExport(data, 'encoder', 'Wcls');
  const { weights, vocab, d_model: dModel, n_heads: nHeads, d_k: dK } = data;
  const wordToIndex = new Map(vocab.map((word, i) => [word, i]));
  const pe = positionalEncoding(MAX_LEN_CLS, dModel);

  return {
    analyze(text) {
      const words = text.trim().split(/\s+/).filter(Boolean).slice(0, MAX_LEN_CLS - 1);
      if (!words.length) throw new Error('Ingrese una oración antes de analizar.');
      const tokens = ['<CLS>', ...words.map(word => wordToIndex.has(word) ? word : '<UNK>')];
      const validLength = tokens.length;
      while (tokens.length < MAX_LEN_CLS) tokens.push('<PAD>');
      const idxs = tokens.map(token => wordToIndex.get(token));
      const keyMask = idxs.map((_, i) => i < validLength);
      const block = transformerBlock(idxs, pe, weights, nHeads, dK, keyMask, false);
      const logits = addBias(matmul([block.encoded[0]], weights.Wcls), weights.bcls)[0];
      const probabilities = softmax(logits);
      return {
        logits,
        probabilities,
        prediction: probabilities[1] > probabilities[0] ? 'POSITIVO' : 'NEGATIVO',
        confidence: Math.max(...probabilities),
        tokens: tokens.slice(0, validLength),
        attention: block.attention.map(map => map.slice(0, validLength).map(row => row.slice(0, validLength)))
      };
    }
  };
}

export function createGPT(data) {
  validateExport(data, 'gpt', 'Wlm');
  const { weights, vocab, d_model: dModel, n_heads: nHeads, d_k: dK } = data;
  const wordToIndex = new Map(vocab.map((word, i) => [word, i]));
  const pe = positionalEncoding(MAX_LEN_GPT, dModel);

  function tokenize(text) {
    const words = text.trim().split(/\s+/).filter(Boolean).slice(0, MAX_LEN_GPT - 2);
    const tokens = ['<BOS>', ...words.map(word => wordToIndex.has(word) ? word : '<UNK>'), '<EOS>'];
    while (tokens.length < MAX_LEN_GPT) tokens.push('<PAD>');
    return { tokens, idxs: tokens.slice(0, MAX_LEN_GPT).map(token => wordToIndex.get(token)) };
  }

  function forwardAll(idxs) {
    const block = transformerBlock(idxs, pe, weights, nHeads, dK, null, true);
    return {
      logits: addBias(matmul(block.encoded, weights.Wlm), weights.blm),
      attention: block.attention
    };
  }

  return {
    forwardAll,
    tokenize,
    generate(seedText, temperature, maxNew = 8, random = Math.random) {
      const cleanSeed = seedText.trim();
      if (!cleanSeed) throw new Error('Ingrese un texto semilla antes de generar.');
      if (!(temperature >= 0.5 && temperature <= 1.5)) throw new Error('La temperatura debe estar entre 0.5 y 1.5.');
      const words = cleanSeed.split(/\s+/);
      for (let step = 0; step < maxNew; step++) {
        const { idxs } = tokenize(words.join(' '));
        const { logits } = forwardAll(idxs);
        const nValid = Math.min(words.length + 1, MAX_LEN_GPT - 1);
        const nextLogits = logits[nValid - 1].map(value => value / temperature);
        nextLogits[wordToIndex.get('<PAD>')] = -Infinity;
        nextLogits[wordToIndex.get('<BOS>')] = -Infinity;
        const probabilities = softmax(nextLogits);
        let threshold = random();
        let nextIndex = probabilities.length - 1;
        for (let i = 0; i < probabilities.length; i++) {
          threshold -= probabilities[i];
          if (threshold <= 0) { nextIndex = i; break; }
        }
        const nextWord = vocab[nextIndex];
        if (nextWord === '<EOS>') break;
        words.push(nextWord);
      }

      const { idxs } = tokenize(words.join(' '));
      const { attention: fullAttention } = forwardAll(idxs);
      const visibleLength = Math.min(words.length + 1, MAX_LEN_GPT - 1);
      const tokens = ['<BOS>', ...words.slice(0, visibleLength - 1).map(word => wordToIndex.has(word) ? word : '<UNK>')];
      return {
        text: words.join(' '),
        tokens,
        attention: fullAttention.map(map => map.slice(0, visibleLength).map(row => row.slice(0, visibleLength)))
      };
    }
  };
}

export async function loadModels(baseUrl = import.meta.url) {
  const urls = [
    new URL('../encoder_weights.json', baseUrl),
    new URL('../gpt_weights.json', baseUrl)
  ];
  const responses = await Promise.all(urls.map(url => fetch(url)));
  for (const response of responses) {
    if (!response.ok) throw new Error(`No se pudieron cargar los pesos (${response.status}). Ejecute el artifact mediante un servidor local.`);
  }
  const [encoderData, gptData] = await Promise.all(responses.map(response => response.json()));
  return { encoder: createEncoder(encoderData), gpt: createGPT(gptData) };
}
