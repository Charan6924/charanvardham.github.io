const article = document.querySelector('#post-content');

Promise.all([
  fetch(new URL('../SDPO.md', window.location.href)).then(response => {
    if (!response.ok) throw new Error(`Markdown request failed (${response.status})`);
    return response.text();
  }),
  new Promise((resolve, reject) => {
    if (window.marked) return resolve();
    const script = document.createElement('script');
    script.src = 'https://cdn.jsdelivr.net/npm/marked/marked.min.js';
    script.onload = resolve;
    script.onerror = () => reject(new Error('Markdown renderer failed to load'));
    document.head.append(script);
  })
]).then(([markdown]) => {
  article.innerHTML = window.marked.parse(markdown, { gfm: true });
  if (window.MathJax) return window.MathJax.typesetPromise([article]);
}).catch(error => {
  article.innerHTML = `<p>${error.message}. Check that this page and <code>SDPO.md</code> are being served from the same site root.</p>`;
});
