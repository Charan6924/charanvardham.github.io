fetch('../SDPO.md').then(response => {
  if (!response.ok) throw new Error('Article could not be loaded');
  return response.text();
}).then(markdown => {
  document.querySelector('#post-content').innerHTML = marked.parse(markdown);
  if (window.MathJax) MathJax.typesetPromise();
}).catch(() => {
  document.querySelector('#post-content').innerHTML = '<p>The article could not be loaded. Please return to the <a href="../index.html">writing archive</a>.</p>';
});
