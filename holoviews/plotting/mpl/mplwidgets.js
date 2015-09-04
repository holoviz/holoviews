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
	if(this.mode == 'nbagg') {
	    this.update_cache();
	    this.update(0);
	    this.set_frame(init_val, 0);
	} else if(this.cached) {
	    this.update_cache();
	    this.update(0);
	} else {
	    this.dynamic_update(0);
	}
    },
    populate_cache : function(idx){
	var cache_id = this.img_id+"_"+idx;
	if(this.load_json) {
	    var data_url = "{{ server }}/" + this.fig_id + "/" + idx;
	    if(this.mode == 'mpld3') {
		$.getJSON(data_url, (function(cache_id) {
		    return function(data) {
			mpld3.draw_figure(cache_id, data);
		    };
		}(cache_id)));
	    } else {
		this.cache[idx].load(data_url);
	    }
	} else {
	    if(this.mode == 'mpld3') {
		mpld3.draw_figure(cache_id, this.frames[idx]);
	    } else {
		this.cache[idx].html(this.frames[idx]);
	    }
	}
    },
    dynamic_update : function(current){
	function callback(msg){
	    /* This callback receives data from Python as a string
	       in order to parse it correctly quotes are sliced off*/
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
	}
	if((this.mode == 'nbagg') || !(current in this.cache)) {
	    var kernel = IPython.notebook.kernel;
	    callbacks = {iopub: {output: $.proxy(callback, this)}};
	    var cmd = "holoviews.plotting.widgets.NdWidget.widgets['" + this.id + "'].update(" + current + ")";
	    kernel.execute("import holoviews;" + cmd, callbacks, {silent : false});
	} else {
	    this.update(current);
	}
    }
}

// Extend MPL widgets with backend specific methods
extend(MPLSelectionWidget.prototype, MPLMethods);
extend(MPLScrubberWidget.prototype, MPLMethods);
