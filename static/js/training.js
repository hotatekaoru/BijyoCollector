/*global $*/
$(function() {
    document.getElementById("output").style.display="none";
    $('#training').click(function() {
        $.ajax({
            url: '/api/training',
            method: 'POST',
            contentType: 'application/json',
            data: "",
            success: function(data) {
                document.getElementById("output").style.display="block";
                img = $('#image');
                img.attr('src', "static/img/retrieve/" + data.results.imgName);
                img.append(img);
                id = $('#imgId');
                id.attr('value', data.results.id);
                id.append(id);
            }
        });
    });

    $('#assort').click(function() {
        var imgId = document.form.imgId.value;
        var rank
        var radioList = document.getElementsByName("rank");
        for(var i=0; i<radioList.length; i++){
            if (radioList[i].checked) {
                rank = radioList[i].value;
                break;
            }
        }
        var json = JSON.stringify({ imgId: imgId, rank: rank});
        $.ajax({
            url: '/api/assort',
            method: 'POST',
            contentType: 'application/json',
            data: json,
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
