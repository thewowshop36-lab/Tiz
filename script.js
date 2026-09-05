document.getElementById('applyForm').addEventListener('submit', function(e) {
    e.preventDefault();
    alert('آپ کی درخواست کامیابی سے موصول ہو گئی ہے!');
});

function copyReferral() {
    navigator.clipboard.writeText(window.location.href);
    alert('ریفریل لنک کاپی ہو گیا ہے!');
}
