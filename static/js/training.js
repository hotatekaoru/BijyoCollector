/*global $*/
$(function() {
    $('#training').click(function() {
        var json = JSON.stringify({ imgNum: $('#imgNum').val()});
        $.ajax({
            url: '/api/training',
            method: 'POST',
            contentType: 'application/json',
            data: json,
            success: function(data) {
               for (var j = 0; j < data.results.length; j++) {
                   img = $('#output tr').eq(j+1).find('td').eq(0).find('img'
                   img.attr('src', "static/img/retrieve/" + data.results[j].imgName);
                   img.append(img);
                   img2 = $('#output tr').eq(j+1).find('td').eq(1).find('img')
                   img2.attr('src', "static/img/trim/" + data.results[j].imgName);
                   img2.append(img2);
                }
            }
        });
    });
});
