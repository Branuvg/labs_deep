import { loadModels } from './model.mjs';

const status = document.querySelector('#model-status');
const sentimentForm = document.querySelector('#sentiment-form');
const gptForm = document.querySelector('#gpt-form');
const temperature = document.querySelector('#temperature');
const temperatureValue = document.querySelector('#temperature-value');

function color(value, accent) {
  const base = accent === 'coral' ? [255, 118, 87] : [199, 244, 100];
  const mix = Math.max(0, Math.min(1, value));
  return `rgb(${Math.round(13 + (base[0] - 13) * mix)}, ${Math.round(14 + (base[1] - 14) * mix)}, ${Math.round(12 + (base[2] - 12) * mix)})`;
}

function drawHeatmap(canvas, values, tokens, accent) {
  const dpr = window.devicePixelRatio || 1;
  const cell = Math.max(25, Math.min(38, Math.floor(480 / tokens.length)));
  const left = 86;
  const top = 100;
  const width = left + cell * tokens.length + 16;
  const height = top + cell * tokens.length + 18;
  canvas.width = width * dpr;
  canvas.height = height * dpr;
  canvas.style.width = `${width}px`;
  canvas.style.height = `${height}px`;
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  ctx.font = '11px ui-monospace, monospace';
  ctx.textBaseline = 'middle';

  values.forEach((row, i) => row.forEach((value, j) => {
    ctx.fillStyle = color(value, accent);
    ctx.fillRect(left + j * cell, top + i * cell, cell - 1, cell - 1);
  }));

  ctx.fillStyle = '#aaa79f';
  tokens.forEach((token, i) => {
    const short = token.length > 10 ? `${token.slice(0, 9)}…` : token;
    ctx.textAlign = 'right';
    ctx.fillText(short, left - 8, top + i * cell + cell / 2);
    ctx.save();
    ctx.translate(left + i * cell + cell / 2, top - 8);
    ctx.rotate(-Math.PI / 3);
    ctx.textAlign = 'left';
    ctx.fillText(short, 0, 0);
    ctx.restore();
  });
  ctx.fillStyle = '#6f7069';
  ctx.font = '10px ui-monospace, monospace';
  ctx.textAlign = 'left';
  ctx.fillText('consulta', 6, top - 14);
  ctx.textAlign = 'right';
  ctx.fillText('clave', width - 16, 14);
}

function renderHeatmaps(container, maps, tokens, accent) {
  container.replaceChildren(...maps.map((map, index) => {
    const card = document.createElement('article');
    card.className = 'heatmap-card';
    const title = document.createElement('h3');
    title.textContent = `Cabeza ${index + 1}`;
    const wrap = document.createElement('div');
    wrap.className = 'canvas-wrap';
    const canvas = document.createElement('canvas');
    canvas.setAttribute('role', 'img');
    canvas.setAttribute('aria-label', `Mapa de atención de la cabeza ${index + 1}`);
    wrap.append(canvas);
    card.append(title, wrap);
    requestAnimationFrame(() => drawHeatmap(canvas, map, tokens, accent));
    return card;
  }));
}

temperature.addEventListener('input', () => { temperatureValue.value = temperature.value; });

try {
  const { encoder, gpt } = await loadModels();
  status.textContent = 'Modelos cargados · inferencia local activa';
  status.classList.remove('loading');
  document.querySelectorAll('button').forEach(button => { button.disabled = false; });

  sentimentForm.addEventListener('submit', event => {
    event.preventDefault();
    const error = document.querySelector('#sentiment-error');
    error.textContent = '';
    try {
      const result = encoder.analyze(document.querySelector('#sentence').value);
      const prediction = document.querySelector('#prediction');
      prediction.className = `prediction ${result.prediction === 'NEGATIVO' ? 'negative' : 'positive'}`;
      prediction.innerHTML = `<strong>${result.prediction}</strong>${(result.confidence * 100).toFixed(1)}% de confianza`;
      renderHeatmaps(document.querySelector('#encoder-heatmaps'), result.attention, result.tokens, 'lime');
    } catch (cause) {
      error.textContent = cause.message;
    }
  });

  gptForm.addEventListener('submit', event => {
    event.preventDefault();
    const error = document.querySelector('#gpt-error');
    error.textContent = '';
    try {
      const result = gpt.generate(document.querySelector('#seed').value, Number(temperature.value));
      document.querySelector('#generated-text').textContent = result.text;
      renderHeatmaps(document.querySelector('#gpt-heatmaps'), result.attention, result.tokens, 'coral');
    } catch (cause) {
      error.textContent = cause.message;
    }
  });
} catch (cause) {
  status.textContent = `Error de carga: ${cause.message}`;
  status.className = 'status failed';
}
