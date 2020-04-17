//Wrap the whole mess in the following function to make it work in Drupal
(function ($) {

  $(document).ready(function () {

    // Original tablecollapse.js functionality
    // Initialize table to fadeOut
    $('.toggle-table').fadeIn();

    //Bind Expand-button method
    $('.collapse-table-button').click(
        function () {
          $('.toggle-table').fadeOut();
        }
    );

    //Bind the toggle button method
    $('.toggle-button').click(
        function () {
          $(this).parent().find("table").fadeToggle("2000", "linear");
        });

    // New code-collapse functionality
    var kCollapseThresholdBytes = 1024;
    $(".highlight-json").each(function (index, elem) {
      // Per Jen's request, here we look into the content to see how big it is,
      // and if it's small, we auto-expanded the div. All .highlight-json divs
      // still have the expand/collapse functionality - we just pre-expand the
      // small ones.
      var $elem = $(elem);
      if ($elem.text().length < kCollapseThresholdBytes)
        $elem.addClass("expanded");

      // Attach the click handler
      $elem.on("click", function (evt) {
        $(evt.target).closest(".highlight-json").toggleClass("expanded");
      });
    });

  });

})(jQuery);
