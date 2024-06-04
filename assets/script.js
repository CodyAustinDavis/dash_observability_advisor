document.addEventListener('DOMContentLoaded', function() {
    const sidebarCol = document.querySelector('.sidebar-col');
    const sidebar = document.querySelector('#sidebar');

    new MutationObserver(function(mutationsList, observer) {
        for(const mutation of mutationsList) {
            if (mutation.attributeName === 'class') {
                if (sidebar.classList.contains('show')) {
                    sidebarCol.classList.remove('collapsed');
                } else {
                    sidebarCol.classList.add('collapsed');
                }
            }
        }
    }).observe(sidebar, { attributes: true });
});