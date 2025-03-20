document.addEventListener("DOMContentLoaded", function() {
    var codeBlocks = document.querySelectorAll("pre.sourceCode");
    codeBlocks.forEach(function(codeBlock) {
    var langTag = document.createElement('p');
    langTag.className = "lang-tag";
    langTag.textContent = codeBlock.className.split(' ')[1];
    codeBlock.parentNode.insertBefore(langTag, codeBlock);
    });
})