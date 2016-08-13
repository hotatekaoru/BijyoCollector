/*global $*/
$(function() {
    $('#test').click(function() {
        $.ajax({
            url: '/api/collect',
            method: 'POST',
            contentType: 'application/json',
            data: "",
            success: function(data) {
               for (var j = 0; j < data.results.length; j++) {
                   img = $('#output tr').eq(j+1).find('td').eq(0).find('img');
                   img.attr('src', "static/img/retrieve/" + data.results[j].imgName);
                   img.append(img);
                   img2 = $('#output tr').eq(j+1).find('td').eq(1).find('img');
                   img2.attr('src', "static/img/trim/" + data.results[j].imgName);
                   img2.append(img2);
                }
            }
        });
    });
});
