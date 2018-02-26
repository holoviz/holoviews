// Define Plotly specific subclasses
function PlotlySelectionWidget() {
  SelectionWidget.apply(this, arguments);
}

function PlotlyScrubberWidget() {
  ScrubberWidget.apply(this, arguments);
}

// Let them inherit from the baseclasses
PlotlySelectionWidget.prototype = Object.create(SelectionWidget.prototype);
PlotlyScrubberWidget.prototype = Object.create(ScrubberWidget.prototype);

// Define methods to override on widgets
var PlotlyMethods = {
  init_slider : function(init_val){
    for (var index in this.frames) {
      this.frames[index] = JSON.parse(this.frames[index]);
    }
  },
  process_msg : function(msg) {
    var data = JSON.parse(msg.content.data);
    this.frames[this.current] = data;
    this.update_cache(true);
    this.update(this.current);
  },
  update : function(current){
    var data = this.frames[current];
    var plot = $('#'+this.id)[0];
    for (var i = 0; i < data.data.length; i++) {
      for (var key in data.data[i]) {
        plot.data[i][key] = data.data[i][key];
      }
    }
    Plotly.relayout(plot, data.layout);
    Plotly.redraw(plot);
  }
}

// Extend Plotly widgets with backend specific methods
extend(PlotlySelectionWidget.prototype, PlotlyMethods);
extend(PlotlyScrubberWidget.prototype, PlotlyMethods);

window.HoloViews.PlotlySelectionWidget = PlotlySelectionWidget;
window.HoloViews.PlotlyScrubberWidget = PlotlyScrubberWidget;
