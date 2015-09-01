var BokehFns = {
    init_slider : function(init_val){
	$.each(this.frames, $.proxy(function(index, frame) {
	    this.frames[index] = JSON.parse(frame);
	}, this));
    },
    update : function(current){
	var data = this.frames[current];

	$.each(data, function(id, value) {
    	    var ds = Bokeh.Collections(value.mode).get(id);
    	    if (ds != undefined) {
    		ds.set(value.data);
    	    }
	});
    },
    dynamic_update : function(current){
	function callback(initialized, msg){
	    /* This callback receives data from Python as a string
	       in order to parse it correctly quotes are sliced off*/
	    if (msg.msg_type == "execute_result") {
		var data = msg.content.data['text/plain'].slice(1, -1);
		this.frames[current] = JSON.parse(data);
		this.update(current);
	    }
	}
	if(!(current in this.frames)) {
	    var kernel = IPython.notebook.kernel;
	    callbacks = {iopub: {output: $.proxy(callback, this, this.initialized)}};
	    var cmd = "holoviews.plotting.widgets.NdWidget.widgets['" + this.id + "'].update(" + current + ")";
	    kernel.execute("import holoviews;" + cmd, callbacks, {silent : false});
	} else {
	    this.update(current);
	}
    }
}

extend(SelectionWidget.prototype, BokehFns);
extend(ScrubberWidget.prototype, BokehFns);
