function HoloViewsWidget() {
}

HoloViewsWidget.prototype.init_slider = function(init_val){
  if(this.load_json) {
    this.from_json()
  } else {
    this.update_cache();
  }
}

HoloViewsWidget.prototype.populate_cache = function(idx){
  this.cache[idx].html(this.frames[idx]);
  if (this.embed) {
    delete this.frames[idx];
  }
}

HoloViewsWidget.prototype.process_error = function(msg){
}

HoloViewsWidget.prototype.from_json = function() {
  var data_url = this.json_path + this.id + '.json';
  $.getJSON(data_url, $.proxy(function(json_data) {
    this.frames = json_data;
    this.update_cache();
    this.update(0);
  }, this));
}

HoloViewsWidget.prototype.dynamic_update = function(current){
  if (current === undefined) {
    return
  }
  if(this.dynamic) {
    current = JSON.stringify(current);
  }
  var that = this
  function callback(msg){
    /* This callback receives data from Python as a string
       in order to parse it correctly quotes are sliced off*/
	if (msg.content.ename != undefined) {
      that.process_error(msg);
    }
    if (msg.msg_type != "execute_result") {
      console.log("Warning: HoloViews callback returned unexpected data for key: (", current, ") with the following content:", msg.content)
    } else {
      if (msg.content.data['text/plain'].includes('Complete')) {
        if (that.queue.length > 0) {
          that.time = Date.now();
          that.dynamic_update(that.queue[that.queue.length-1]);
          that.queue = [];
        } else {
          that.wait = false;
        }
        return
      }
    }
  }
  this.current = current;
  if ((window.Jupyter !== undefined) && (Jupyter.notebook.kernel != null)) {
    var kernel = Jupyter.notebook.kernel;
    var callbacks = {iopub: {output: callback}};
    var cmd = "holoviews.plotting.widgets.NdWidget.widgets['" + this.id + "'].update(" + current + ")";
    kernel.execute("import holoviews;" + cmd, callbacks, {silent : false});
  }
}

HoloViewsWidget.prototype.update_cache = function(force){
  var frame_len = Object.keys(this.frames).length;
  for (var i=0; i<frame_len; i++) {
    if(!this.load_json || this.dynamic)  {
      var frame = Object.keys(this.frames)[i];
    } else {
      var frame = i;
    }
    if(!(frame in this.cache) || force) {
      if ((frame in this.cache) && force) { this.cache[frame].remove() }
      this.cache[frame] = $('<div />').appendTo("#"+"_anim_img"+this.id).hide();
      var cache_id = "_anim_img"+this.id+"_"+frame;
      this.cache[frame].attr("id", cache_id);
      this.populate_cache(frame);
    }
  }
}

HoloViewsWidget.prototype.update = function(current){
  if(current in this.cache) {
    $.each(this.cache, function(index, value) {
      value.hide();
    });
    this.cache[current].show();
    this.wait = false;
  }
}

HoloViewsWidget.prototype.init_comms = function() {
  if ((window.Jupyter !== undefined) && (Jupyter.notebook.kernel !== undefined)) {
    var widget = this;
    comm_manager = Jupyter.notebook.kernel.comm_manager;
    comm_manager.register_target(this.id, function (comm) {
      comm.on_msg(function (msg) { widget.process_msg(msg) });
    });
  }
}

HoloViewsWidget.prototype.process_msg = function(msg) {
}

function SelectionWidget(frames, id, slider_ids, keyMap, dim_vals, notFound, load_json, mode, cached, json_path, dynamic){
  this.frames = frames;
  this.id = id;
  this.slider_ids = slider_ids;
  this.keyMap = keyMap
  this.current_frame = 0;
  this.current_vals = dim_vals;
  this.load_json = load_json;
  this.mode = mode;
  this.notFound = notFound;
  this.cached = cached;
  this.dynamic = dynamic;
  this.cache = {};
  this.json_path = json_path;
  this.init_slider(this.current_vals[0]);
  this.queue = [];
  this.wait = false;
  if (!this.cached || this.dynamic) {
    this.init_comms()
  }
}

SelectionWidget.prototype = new HoloViewsWidget;


SelectionWidget.prototype.get_key = function(current_vals) {
  var key = "(";
  for (var i=0; i<this.slider_ids.length; i++)
  {
    let val = this.current_vals[i];
	if (!(typeof val === 'string')) {
      if (val % 1 === 0) { val = val.toFixed(1); }
      else { val = val.toFixed(10); val = val.slice(0, val.length-1);}
    }
    key += "'" + val + "'";
    if(i != this.slider_ids.length-1) { key += ', ';}
    else if(this.slider_ids.length == 1) { key += ',';}
  }
  key += ")";
  return this.keyMap[key];
}

SelectionWidget.prototype.set_frame = function(dim_val, dim_idx){
  this.current_vals[dim_idx] = dim_val;
  var key = this.current_vals;
  if (!this.dynamic) {
    key = this.get_key(key)
  }
  if (this.dynamic || !this.cached) {
    if ((this.time !== undefined) && ((this.wait) && ((this.time + 10000) > Date.now()))) {
      this.queue.push(key);
      return
    }
    this.queue = [];
    this.time = Date.now();
    this.current_frame = key;
    this.wait = true;
    this.dynamic_update(key)
  } else if (key !== undefined) {
    this.update(key)
  }
}


/* Define the ScrubberWidget class */
function ScrubberWidget(frames, num_frames, id, interval, load_json, mode, cached, json_path, dynamic){
  this.slider_id = "_anim_slider" + id;
  this.loop_select_id = "_anim_loop_select" + id;
  this.id = id;
  this.interval = interval;
  this.current_frame = 0;
  this.direction = 0;
  this.dynamic = dynamic;
  this.timer = null;
  this.load_json = load_json;
  this.mode = mode;
  this.cached = cached;
  this.frames = frames;
  this.cache = {};
  this.length = num_frames;
  this.json_path = json_path;
  document.getElementById(this.slider_id).max = this.length - 1;
  this.init_slider(0);
  this.wait = false;
  this.queue = [];
  if (!this.cached || this.dynamic) {
    this.init_comms()
  }
}

ScrubberWidget.prototype = new HoloViewsWidget;

ScrubberWidget.prototype.set_frame = function(frame){
  this.current_frame = frame;
  var widget = document.getElementById(this.slider_id);
  if (widget === null) {
    this.pause_animation();
    return
  }
  widget.value = this.current_frame;
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
  if(this.current_frame < this.length - 1){
    this.next_frame();
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
  if(this.current_frame > 0){
    this.previous_frame();
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

function update_widget(widget, values) {
  if (widget.hasClass("ui-slider")) {
    widget.slider('option', {
      min: 0,
      max: values.length-1,
      dim_vals: values,
      value: 0,
      dim_labels: values
	})
    widget.slider('option', 'slide').call(widget, event, {value: 0})
  } else {
    widget.empty();
    for (var i=0; i<values.length; i++){
      widget.append($("<option>", {
        value: i,
        text: values[i]
      }))
    };
    widget.data('values', values);
    widget.data('value', 0);
    widget.trigger("change");
  };
}

function init_slider(id, dim, values, next_vals, labels, dynamic, step, next_dim,
                     dim_idx, jQueryUI_CDN, UNDERSCORE_CDN) {
	// Slider JS Block START
  function loadcssfile(filename){
    var fileref=document.createElement("link")
    fileref.setAttribute("rel", "stylesheet")
    fileref.setAttribute("type", "text/css")
    fileref.setAttribute("href", filename)
    document.getElementsByTagName("head")[0].appendChild(fileref)
  }
  loadcssfile("https://code.jquery.com/ui/1.10.4/themes/smoothness/jquery-ui.css");
  /* Check if jQuery and jQueryUI have been loaded
     otherwise load with require.js */
  var jQuery = window.jQuery,
    // check for old versions of jQuery
    oldjQuery = jQuery && !!jQuery.fn.jquery.match(/^1\.[0-4](\.|$)/),
    jquery_path = '',
    paths = {},
    noConflict;
  var jQueryUI = jQuery.ui;
  // check for jQuery
  if (!jQuery || oldjQuery) {
    // load if it's not available or doesn't meet min standards
    paths.jQuery = jQuery;
    noConflict = !!oldjQuery;
  } else {
    // register the current jQuery
    define('jquery', [], function() { return jQuery; });
  }
  if (!jQueryUI) {
	paths.jQueryUI = jQueryUI_CDN.slice(null, -3);
  } else {
    define('jQueryUI', [], function() { return jQuery.ui; });
  }
  paths.underscore = UNDERSCORE_CDN.slice(null, -3);
  var jquery_require = {
    paths: paths,
    shim: {
      "jQueryUI": {
        exports:"$",
        deps: ['jquery']
      },
      "underscore": {
        exports: '_'
      }
    }
  }
  require.config(jquery_require);
  require(["jQueryUI", "underscore"], function(jUI, _){
    if (noConflict) $.noConflict(true);
    var vals = values;
    if (dynamic && vals.constructor === Array) {
      var min = parseFloat(vals[0]);
      var max = parseFloat(vals[vals.length-1]);
      var step = step;
      var wlabels = [min];
    } else {
      var min = 0;
      if (dynamic) {
        var max = Object.keys(vals).length - 1;
      } else {
        var max = vals.length - 1;
      }
      var step = 1;
      var wlabels = labels;
    }
    function adjustFontSize(text) {
      var width_ratio = (text.parent().width()/8)/text.val().length;
      var size = Math.min(0.9, Math.max(0.6, width_ratio))+'em';
      text.css('font-size', size);
    }
    var slider = $('#_anim_widget'+id+'_'+dim);
    slider.slider({
      animate: "fast",
      min: min,
      max: max,
      step: step,
      value: min,
      dim_vals: vals,
      dim_labels: wlabels,
      next_vals: next_vals,
      slide: function(event, ui) {
        var vals = slider.slider("option", "dim_vals");
        var next_vals = slider.slider("option", "next_vals");
        var dlabels = slider.slider("option", "dim_labels");
        if (dynamic) {
          var dim_val = ui.value;
          if (vals.constructor === Array) {
            var label = ui.value;
          } else {
            var label = dlabels[ui.value];
          }
        } else {
          var dim_val = vals[ui.value];
          var label = dlabels[ui.value];
        }
        var text = $('#textInput'+id+'_'+dim);
        text.val(label);
        adjustFontSize(text);
        HoloViews.index[id].set_frame(dim_val, dim_idx);
        if (Object.keys(next_vals).length > 0) {
          var new_vals = next_vals[dim_val];
          var next_widget = $('#_anim_widget'+id+'_'+next_dim);
          update_widget(next_widget, new_vals);
        }
      }
    });
    slider.keypress(function(event) {
      if (event.which == 80 || event.which == 112) {
        var start = slider.slider("option", "value");
        var stop =  slider.slider("option", "max");
        for (var i=start; i<=stop; i++) {
          var delay = i*delay;
          $.proxy(function doSetTimeout(i) { setTimeout($.proxy(function() {
            var val = {value:i};
            slider.slider('value',i);
            slider.slider("option", "slide")(null, val);
          }, slider), delay);}, slider)(i);
        }
      }
      if (event.which == 82 || event.which == 114) {
        var start = slider.slider("option", "value");
        var stop =  slider.slider("option", "min");
        var count = 0;
        for (var i=start; i>=stop; i--) {
          var delay = count*delay;
          count = count + 1;
          $.proxy(function doSetTimeout(i) { setTimeout($.proxy(function() {
            var val = {value:i};
            slider.slider('value',i);
            slider.slider("option", "slide")(null, val);
          }, slider), delay);}, slider)(i);
        }
      }
    });
    var textInput = $('#textInput'+id+'_'+dim)
    textInput.val(labels[0]);
    adjustFontSize(textInput);
  });
}

function init_dropdown(id, dim, vals, next_vals, labels, next_dim, dim_idx, dynamic) {
  var widget = $("#_anim_widget"+id+'_'+dim);
  widget.data('values', vals)
  for (var i=0; i<vals.length; i++){
    if (dynamic) {
      var val = vals[i];
    } else {
      var val = i;
    }
    widget.append($("<option>", {
      value: val,
      text: labels[i]
    }));
  };
  widget.data("next_vals", next_vals);
  widget.on('change', function(event, ui) {
    if (dynamic) {
      var dim_val = parseInt(this.value);
    } else {
      var dim_val = $.data(this, 'values')[this.value];
    }
    var next_vals = $.data(this, "next_vals");
    if (Object.keys(next_vals).length > 0) {
      var new_vals = next_vals[dim_val];
      var next_widget = $('#_anim_widget'+id+'_'+next_dim);
      update_widget(next_widget, new_vals);
    }
	var widgets = HoloViews.index[id]
    if (widgets) {
      widgets.set_frame(dim_val, dim_idx);
    }
  });
}

if (!window.HoloViews) {
  window.HoloViews = {
    ScrubberWidget: ScrubberWidget,
    SelectionWidget: SelectionWidget,
    update_widget: update_widget,
    init_slider: init_slider,
    init_dropdown: init_dropdown,
    comms: {},
    comm_state: {},
    index: {}
  }
}
