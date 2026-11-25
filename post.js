const article = document.querySelector('#post-content');
const markdownPath = article.dataset.markdown;

fetch(new URL(markdownPath, window.location.href)).then(response => {
  if (!response.ok) throw new Error(`Markdown request failed (${response.status})`);
  return response.text();
}).then(markdown => {
  if (!window.marked) throw new Error('Markdown renderer failed to load');
  const expressions = [];
  const protectedMarkdown = markdown.replace(/\$\$[\s\S]*?\$\$|\$(?!\$)[^\n$]*\$/g, expression => {
    const token = `MATHTOKEN${expressions.length}END`;
    expressions.push(expression.replace(/\n> ?/g, '\n'));
    return token;
  });
  let html = window.marked.parse(protectedMarkdown, { gfm: true });
  html = html.replace(/<blockquote>\s*<p>\[!note\]-?\s+([^\n<]+)\n([\s\S]*?)<\/blockquote>/g,
    (_, title, body) => `<aside class="note"><strong>${title}</strong><p>${body}</aside>`);
  html = html.replace(/MATHTOKEN(\d+)END/g, (_, index) => expressions[Number(index)]
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;'));
  article.innerHTML = html;
  if (window.MathJax) return window.MathJax.typesetPromise([article]);
}).catch(error => {
  article.textContent = `Unable to load the article: ${error.message}`;
});
