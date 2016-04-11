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
		var cache_id = this.img_id+"_"+idx;
		if(this.mode == 'mpld3') {
			mpld3.draw_figure(cache_id, this.frames[idx]);
		} else {
			this.cache[idx].html(this.frames[idx]);
		}
		if (this.embed) {
			delete this.frames[idx];
		}
	},
	dynamic_update : function(current){
		if (this.dynamic) {
			current = JSON.stringify(current);
		}
		function callback(msg){
			/* This callback receives data from Python as a string
			 in order to parse it correctly quotes are sliced off*/
			if (msg.content.ename != undefined) {
				this.process_error(msg);
			}
			if (msg.msg_type != "execute_result") {
				console.log("Warning: HoloViews callback returned unexpected data for key: (", current, ") with the following content:", msg.content)
				this.time = undefined;
				return
			}
			if (!(this.mode == 'nbagg')) {
				if(!(current in this.cache)) {
					var data = msg.content.data['text/plain'].slice(1, -1);
					if(this.mode == 'mpld3'){
						data = JSON.parse(data)[0];
					}
					this.frames[current] = data;
					this.update_cache();
				}
				this.update(current);
			}
			this.timed = (Date.now() - this.time) * 1.5;
			this.wait = false;
			if (this.queue.length > 0) {
				var current_vals = this.queue[this.queue.length-1];
				this.time = Date.now();
				this.dynamic_update(current_vals);
				this.queue = [];
			}
		}
		var kernel = IPython.notebook.kernel;
		callbacks = {iopub: {output: $.proxy(callback, this)}};
		var cmd = "holoviews.plotting.widgets.NdWidget.widgets['" + this.id + "'].update(" + current + ")";
		kernel.execute("import holoviews;" + cmd, callbacks, {silent : false});
	}
}

// Extend MPL widgets with backend specific methods
extend(MPLSelectionWidget.prototype, MPLMethods);
extend(MPLScrubberWidget.prototype, MPLMethods);
