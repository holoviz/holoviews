// Define Bokeh specific subclasses
function BokehSelectionWidget() {
	SelectionWidget.apply(this, arguments);
}

function BokehScrubberWidget() {
	ScrubberWidget.apply(this, arguments);
}

// Let them inherit from the baseclasses
BokehSelectionWidget.prototype = Object.create(SelectionWidget.prototype);
BokehScrubberWidget.prototype = Object.create(ScrubberWidget.prototype);

// Define methods to override on widgets
var BokehMethods = {
	update_cache : function(){
		$.each(this.frames, $.proxy(function(index, frame) {
			this.frames[index] = JSON.parse(frame);
		}, this));
	},
	update : function(current){
		if (current === undefined) {
			var data = undefined;
		} else {
			var data = this.frames[current];
		}
		if (data !== undefined) {
			var doc = Bokeh.index[data.root].model.document;
			doc.apply_json_patch(data.content);
		}
	},
	init_comms : function() {
	}
}

// Extend Bokeh widgets with backend specific methods
extend(BokehSelectionWidget.prototype, BokehMethods);
extend(BokehScrubberWidget.prototype, BokehMethods);
