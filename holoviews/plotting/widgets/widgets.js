function HoloViewsWidget(){
}

HoloViewsWidget.prototype.init_slider = function(init_val){
    if(this.cached) {
	this.update_cache();
	this.update(0);
    } else {
	this.dynamic_update(0);
    }
}

HoloViewsWidget.prototype.populate_cache = function(idx){
    if(this.load_json) {
	var data_url = this.server + this.fig_id + "/" + idx;
	this.cache[idx].load(data_url);
    } else {
	this.cache[idx].html(this.frames[idx]);
	if (this.embed) {
	    delete this.frames[idx];
	}
    }
}

HoloViewsWidget.prototype.dynamic_update = function(current){
    function callback(msg){
	/* This callback receives data from Python as a string
	   in order to parse it correctly quotes are sliced off*/
	var data = msg.content.data['text/plain'].slice(1, -1);
	this.frames[current] = data;
	this.update_cache();
	this.update(current);
    }
    if(!(current in this.cache)) {
	var kernel = IPython.notebook.kernel;
	callbacks = {iopub: {output: $.proxy(callback, this)}};
	var cmd = "holoviews.plotting.widgets.NdWidget.widgets['" + this.id + "'].update(" + current + ")";
	kernel.execute("import holoviews;" + cmd, callbacks, {silent : false});
    } else {
	this.update(current);
    }
}

HoloViewsWidget.prototype.update_cache = function(){
    if(this.load_json) {
	var frame_len = Object.keys(this.keyMap).length;
    } else {
	var frame_len = Object.keys(this.frames).length;
    }
    for (var i=0; i<frame_len; i++) {
	if(!this.load_json) {
	    i = Object.keys(this.frames)[i];
	}
	if(!(i in this.cache)) {
	    this.cache[i] = $('<div />').appendTo("#" + this.img_id).hide();
	    var cache_id = this.img_id+"_"+i;
	    this.cache[i].attr("id", cache_id);
	    this.populate_cache(i);
	}
    }
}

HoloViewsWidget.prototype.update = function(current){
    if(current in this.cache) {
	$.each(this.cache, function(index, value) {
	    value.hide();
	});
	this.cache[current].show();
    }
}


function SelectionWidget(frames, id, slider_ids, keyMap, dim_vals, notFound, load_json, mode, cached, server){
    this.frames = frames;
    this.fig_id = "fig_" + id;
    this.img_id = "_anim_img" + id;
    this.id = id;
    this.server = server;
    this.slider_ids = slider_ids;
    this.keyMap = keyMap
    this.current_frame = 0;
    this.current_vals = dim_vals;
    this.load_json = load_json;
    this.mode = mode;
    this.notFound = notFound;
    this.cached = cached;
    this.cache = {};
    this.init_slider(this.current_vals[0]);
}

SelectionWidget.prototype = new HoloViewsWidget;

SelectionWidget.prototype.set_frame = function(dim_val, dim_idx){
    this.current_vals[dim_idx] = dim_val;
    var key = "(";
    for (var i=0; i<this.slider_ids.length; i++)
    {
        val = this.current_vals[i];
        if (!(_.isString(val))) {
            if (val % 1 === 0) { var fixed = 1;}
            else { var fixed = 10;}
            val = val.toFixed(fixed)
        }
        key += "'" + val + "'";
        if(i != this.slider_ids.length-1) { key += ', ';}
        else if(this.slider_ids.length == 1) { key += ',';}
    }
    key += ")";
    var current = this.keyMap[key];
    this.current_frame = current;
    if(this.cached) {
	this.update(current)
    } else {
        this.dynamic_update(current)
    }
}


/* Define the ScrubberWidget class */
function ScrubberWidget(frames, num_frames, id, interval, load_json, mode, cached){
    this.img_id = "_anim_img" + id;
    this.slider_id = "_anim_slider" + id;
    this.loop_select_id = "_anim_loop_select" + id;
    this.id = id;
    this.fig_id = "fig_" + id;
    this.interval = interval;
    this.current_frame = 0;
    this.direction = 0;
    this.timer = null;
    this.load_json = load_json;
    this.mode = mode;
    this.cached = cached;
    this.frames = frames;
    this.cache = {};
    this.length = num_frames;
    document.getElementById(this.slider_id).max = this.length - 1;
    this.init_slider(0);
}

ScrubberWidget.prototype = new HoloViewsWidget;

ScrubberWidget.prototype.set_frame = function(frame){
    this.current_frame = frame;
    document.getElementById(this.slider_id).value = this.current_frame;
    if(this.cached) {
	this.update(frame)
    } else {
        this.dynamic_update(frame)
    }
}

ScrubberWidget.prototype.get_loop_state = function(){
    var button_group = document[this.loop_select_id].state;
    for (var i = 0; i < button_group.length; i++) {
        var button = button_group[i];
        if (button.checked) {
            return button.value;
        }
    }
    return undefined;
}

ScrubberWidget.prototype.update = function(current){
    if(current in this.cache) {
	$.each(this.cache, function(index, value) {
	    value.hide();
	});
	this.cache[current].show();
    }
}

ScrubberWidget.prototype.next_frame = function() {
    this.set_frame(Math.min(this.length - 1, this.current_frame + 1));
}

ScrubberWidget.prototype.previous_frame = function() {
    this.set_frame(Math.max(0, this.current_frame - 1));
}

ScrubberWidget.prototype.first_frame = function() {
    this.set_frame(0);
}

ScrubberWidget.prototype.last_frame = function() {
    this.set_frame(this.length - 1);
}

ScrubberWidget.prototype.slower = function() {
    this.interval /= 0.7;
    if(this.direction > 0){this.play_animation();}
    else if(this.direction < 0){this.reverse_animation();}
}

ScrubberWidget.prototype.faster = function() {
    this.interval *= 0.7;
    if(this.direction > 0){this.play_animation();}
    else if(this.direction < 0){this.reverse_animation();}
}

ScrubberWidget.prototype.anim_step_forward = function() {
    this.current_frame += 1;
    if(this.current_frame < this.length){
        this.set_frame(this.current_frame);
    }else{
        var loop_state = this.get_loop_state();
        if(loop_state == "loop"){
            this.first_frame();
        }else if(loop_state == "reflect"){
            this.last_frame();
            this.reverse_animation();
        }else{
            this.pause_animation();
            this.last_frame();
        }
    }
}

ScrubberWidget.prototype.anim_step_reverse = function() {
    this.current_frame -= 1;
    if(this.current_frame >= 0){
        this.set_frame(this.current_frame);
    } else {
        var loop_state = this.get_loop_state();
        if(loop_state == "loop"){
            this.last_frame();
        }else if(loop_state == "reflect"){
            this.first_frame();
            this.play_animation();
        }else{
            this.pause_animation();
            this.first_frame();
        }
    }
}

ScrubberWidget.prototype.pause_animation = function() {
    this.direction = 0;
    if (this.timer){
        clearInterval(this.timer);
        this.timer = null;
    }
}

ScrubberWidget.prototype.play_animation = function() {
    this.pause_animation();
    this.direction = 1;
    var t = this;
    if (!this.timer) this.timer = setInterval(function(){t.anim_step_forward();}, this.interval);
}

ScrubberWidget.prototype.reverse_animation = function() {
    this.pause_animation();
    this.direction = -1;
    var t = this;
    if (!this.timer) this.timer = setInterval(function(){t.anim_step_reverse();}, this.interval);
}

function extend(destination, source) {
  for (var k in source) {
    if (source.hasOwnProperty(k)) {
      destination[k] = source[k];
    }
  }
  return destination;
}
