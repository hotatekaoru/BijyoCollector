/*global $*/
$(function() {
    $('#training').click(function() {
        var imgNum = $('#imgNum').val();
        $.ajax({
            url: '/api/training',
            method: 'POST',
            contentType: 'application/json',
            data: $('form').serialize(),
            success: function(data) {
                var element = document.createElement('div');
                element.id = "id";
                element.innerHTML = "hogehoge";
                element.style.backgroundColor = 'red';
                var objBody = document.getElementsByTagName("body").item(0);
                objBody.appendChild(element);
            }
        });
    });
});
