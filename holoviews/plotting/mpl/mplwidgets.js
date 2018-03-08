// Define MPL specific subclasses
function MPLSelectionWidget() {
  SelectionWidget.apply(this, arguments);
}

function MPLScrubberWidget() {
  ScrubberWidget.apply(this, arguments);
}

// Let them inherit from the baseclasses
MPLSelectionWidget.prototype = Object.create(SelectionWidget.prototype);
MPLScrubberWidget.prototype = Object.create(ScrubberWidget.prototype);

// Define methods to override on widgets
var MPLMethods = {
  init_slider : function(init_val){
    if(this.load_json) {
      this.from_json()
    } else {
      this.update_cache();
    }
    if (this.dynamic | !this.cached | (this.current_vals === undefined)) {
      this.update(0)
    } else {
      this.set_frame(this.current_vals[0], 0)
    }
  },
  process_msg : function(msg) {
    var data = msg.content.data;
    this.frames[this.current] = data;
    this.update_cache(true);
    this.update(this.current);
  }
}
// Extend MPL widgets with backend specific methods
extend(MPLSelectionWidget.prototype, MPLMethods);
extend(MPLScrubberWidget.prototype, MPLMethods);

window.HoloViews.MPLSelectionWidget = MPLSelectionWidget
window.HoloViews.MPLScrubberWidget = MPLScrubberWidget
