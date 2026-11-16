const article = document.querySelector('#post-content');

fetch(new URL('../SDPO.md', window.location.href)).then(response => {
  if (!response.ok) throw new Error(`Markdown request failed (${response.status})`);
  return response.text();
}).then(markdown => {
  if (!window.marked) throw new Error('Markdown renderer failed to load');
  const expressions = [];
  const protectedMarkdown = markdown.replace(/\$\$[\s\S]*?\$\$|\$(?!\$)[^\n$]*\$/g, expression => {
    const token = `SDPOMATHTOKEN${expressions.length}END`;
    expressions.push(expression);
    return token;
  });
  let html = window.marked.parse(protectedMarkdown, { gfm: true });
  html = html.replace(/SDPOMATHTOKEN(\d+)END/g, (_, index) => expressions[Number(index)]
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;'));
  article.innerHTML = html;
  if (window.MathJax) return window.MathJax.typesetPromise([article]);
}).catch(error => {
  article.innerHTML = `<p>${error.message}. Check that this page and <code>SDPO.md</code> are being served from the same site root.</p>`;
});
