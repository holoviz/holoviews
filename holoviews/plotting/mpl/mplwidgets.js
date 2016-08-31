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
		this.update(0);
		if(this.mode == 'nbagg') {
			this.set_frame(init_val, 0);
		}
	},
	populate_cache : function(idx){
		var cache_id = "_anim_img"+this.id+"_"+idx;
		if(this.mode == 'mpld3') {
			mpld3.draw_figure(cache_id, this.frames[idx]);
		} else {
			this.cache[idx].html(this.frames[idx]);
		}
		if (this.embed) {
			delete this.frames[idx];
		}
	},
	process_msg : function(msg) {
		if (!(this.mode == 'nbagg')) {
			var data = msg.content.data;
			this.frames[this.current] = data;
			this.update_cache(true);
			this.update(this.current);
		}
	}
}
// Extend MPL widgets with backend specific methods
extend(MPLSelectionWidget.prototype, MPLMethods);
extend(MPLScrubberWidget.prototype, MPLMethods);
