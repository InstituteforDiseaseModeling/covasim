$(document).ready(function () {
    //----------------------------------------------------------------------------
    // Adapted from highlight v5
    // IDM version given here takes className as an argument.
    // Highlights arbitrary terms.
    // <http://johannburkard.de/blog/programming/javascript/highlight-javascript-text-higlighting-jquery-plugin.html>
    // MIT license.
    // Johann Burkard
    // <http://johannburkard.de>
    // <mailto:jb@eaio.com>
    //----------------------------------------------------------------------------
    jQuery.fn.highlight = function (pat, className) {
        function innerHighlight(node, pat) {
            var skip = 0;
            if (node.nodeType == 3) {
                var pos = node.data.toUpperCase().indexOf(pat);
                pos -= (node.data.substr(0, pos).toUpperCase().length -
                    node.data.substr(0, pos).length);
                if (pos >= 0) {
                    var spanNode = document.createElement('span');
                    spanNode.className = className;
                    var middleBit = node.splitText(pos);
                    middleBit.splitText(pat.length);
                    var middleClone = middleBit.cloneNode(true);
                    spanNode.appendChild(middleClone);
                    middleBit.parentNode.replaceChild(spanNode, middleBit);
                    skip = 1;
                }
            } else if (node.nodeType == 1 && node.childNodes &&
                !/(script|style)/i.test(node.tagName)) {
                for (var i = 0; i < node.childNodes.length; ++i)
                    i += innerHighlight(node.childNodes[i], pat);
            }
            return skip;
        }

        return this.length && pat && pat.length ? this.each(function () {
            innerHighlight(this, pat.toUpperCase());
        }) : this;
    };

    //----------------------------------------------------------------------------
    jQuery.fn.removeHighlight = function (className) {
        return this.find("span." + className).each(
            function () {
                var parentNode = this.parentNode;
                // Noop $(this)[0] is there to circumvent a warning
                parentNode.replaceChild(this.firstChild, $(this)[0]);
                parentNode.normalize();
            });
    };

    //----------------------------------------------------------------------------
    function getParameterByName(name, url) {
        if (!url) {
            url = window.location.href;
        }
        name = name.replace(/[\[\]]/g, "\\$&");
        var regex = new RegExp("[?&]" + name + "(=([^&#]*)|&|#|$)"),
            results = regex.exec(url);
        if (!results) return null;
        if (!results[2]) return '';
        return decodeURIComponent(results[2].replace(/\+/g, " "));
    }

    // Connect tipue search to the search box
    $('#tipue_search_input').tipuesearch({showTime: false, wholeWords: false});

    // Process a "searchText=xxx" param on the URL and use highlight if present
    {
        var searchText = getParameterByName("searchText");
        if (searchText) {
            $("div.document").highlight(searchText, "search-highlight");
        }
    }
});
