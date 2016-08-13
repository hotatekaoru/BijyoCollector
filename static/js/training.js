/*global $*/
$(function() {
    $('#training').click(function() {
        /*var json = JSON.stringify({ imgNum: $('#imgNum').val()});*/
        $.ajax({
            url: '/api/training',
            method: 'POST',
            contentType: 'application/json',
            data: "",
            success: function(data) {
                img = $('#image');
                img.attr('src', "static/img/retrieve/" + data.results.imgName);
                img.append(img);
                id = $('#imgId');
                id.attr('value', data.results.id);
                id.append(id);
            }
        });
    });
});
