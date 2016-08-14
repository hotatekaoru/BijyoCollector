/*global $*/
$(function() {
    document.getElementById("output").style.display="none";
    $('#collect').click(function() {
        $.ajax({
            url: '/api/collect',
            method: 'POST',
            contentType: 'application/json',
            data: "",
            success: function(data) {
                document.getElementById("output").style.display="block";
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
    $('#train').click(function() {
        window.location.href = '/training';
    });
    $('#test').click(function() {
        window.location.href = '/test';
    });
});
