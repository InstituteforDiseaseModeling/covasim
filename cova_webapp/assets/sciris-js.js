(function(f){if(typeof exports==="object"&&typeof module!=="undefined"){module.exports=f()}else if(typeof define==="function"&&define.amd){define([],f)}else{var g;if(typeof window!=="undefined"){g=window}else if(typeof global!=="undefined"){g=global}else if(typeof self!=="undefined"){g=self}else{g=this}g.sciris = f()}})(function(){var define,module,exports;return (function(){function r(e,n,t){function o(i,f){if(!n[i]){if(!e[i]){var c="function"==typeof require&&require;if(!f&&c)return c(i,!0);if(u)return u(i,!0);var a=new Error("Cannot find module '"+i+"'");throw a.code="MODULE_NOT_FOUND",a}var p=n[i]={exports:{}};e[i][0].call(p.exports,function(r){var n=e[i][1][r];return o(n||r)},p,p.exports,r,e,n,t)}return n[i].exports}for(var u="function"==typeof require&&require,i=0;i<t.length;i++)o(t[i]);return o}return r})()({1:[function(require,module,exports){
(function (process,global,setImmediate){
/*!
 * sciris-js v0.2.18
 * (c) 2019-present Sciris <info@sciris.org>
 * Released under the MIT License.
 */
(function (global, factory) {
  typeof exports === 'object' && typeof module !== 'undefined' ? factory(exports) :
  typeof define === 'function' && define.amd ? define(['exports'], factory) :
  (factory((global['sciris-js'] = {})));
}(this, (function (exports) { 'use strict';

  /*!
   * Vue.js v2.6.10
   * (c) 2014-2019 Evan You
   * Released under the MIT License.
   */
  /*  */

  var emptyObject = Object.freeze({});

  // These helpers produce better VM code in JS engines due to their
  // explicitness and function inlining.
  function isUndef (v) {
    return v === undefined || v === null
  }

  function isDef (v) {
    return v !== undefined && v !== null
  }

  function isTrue (v) {
    return v === true
  }

  function isFalse (v) {
    return v === false
  }

  /**
   * Check if value is primitive.
   */
  function isPrimitive (value) {
    return (
      typeof value === 'string' ||
      typeof value === 'number' ||
      // $flow-disable-line
      typeof value === 'symbol' ||
      typeof value === 'boolean'
    )
  }

  /**
   * Quick object check - this is primarily used to tell
   * Objects from primitive values when we know the value
   * is a JSON-compliant type.
   */
  function isObject (obj) {
    return obj !== null && typeof obj === 'object'
  }

  /**
   * Get the raw type string of a value, e.g., [object Object].
   */
  var _toString = Object.prototype.toString;

  function toRawType (value) {
    return _toString.call(value).slice(8, -1)
  }

  /**
   * Strict object type check. Only returns true
   * for plain JavaScript objects.
   */
  function isPlainObject (obj) {
    return _toString.call(obj) === '[object Object]'
  }

  function isRegExp (v) {
    return _toString.call(v) === '[object RegExp]'
  }

  /**
   * Check if val is a valid array index.
   */
  function isValidArrayIndex (val) {
    var n = parseFloat(String(val));
    return n >= 0 && Math.floor(n) === n && isFinite(val)
  }

  function isPromise (val) {
    return (
      isDef(val) &&
      typeof val.then === 'function' &&
      typeof val.catch === 'function'
    )
  }

  /**
   * Convert a value to a string that is actually rendered.
   */
  function toString (val) {
    return val == null
      ? ''
      : Array.isArray(val) || (isPlainObject(val) && val.toString === _toString)
        ? JSON.stringify(val, null, 2)
        : String(val)
  }

  /**
   * Convert an input value to a number for persistence.
   * If the conversion fails, return original string.
   */
  function toNumber (val) {
    var n = parseFloat(val);
    return isNaN(n) ? val : n
  }

  /**
   * Make a map and return a function for checking if a key
   * is in that map.
   */
  function makeMap (
    str,
    expectsLowerCase
  ) {
    var map = Object.create(null);
    var list = str.split(',');
    for (var i = 0; i < list.length; i++) {
      map[list[i]] = true;
    }
    return expectsLowerCase
      ? function (val) { return map[val.toLowerCase()]; }
      : function (val) { return map[val]; }
  }

  /**
   * Check if a tag is a built-in tag.
   */
  var isBuiltInTag = makeMap('slot,component', true);

  /**
   * Check if an attribute is a reserved attribute.
   */
  var isReservedAttribute = makeMap('key,ref,slot,slot-scope,is');

  /**
   * Remove an item from an array.
   */
  function remove (arr, item) {
    if (arr.length) {
      var index = arr.indexOf(item);
      if (index > -1) {
        return arr.splice(index, 1)
      }
    }
  }

  /**
   * Check whether an object has the property.
   */
  var hasOwnProperty = Object.prototype.hasOwnProperty;
  function hasOwn (obj, key) {
    return hasOwnProperty.call(obj, key)
  }

  /**
   * Create a cached version of a pure function.
   */
  function cached (fn) {
    var cache = Object.create(null);
    return (function cachedFn (str) {
      var hit = cache[str];
      return hit || (cache[str] = fn(str))
    })
  }

  /**
   * Camelize a hyphen-delimited string.
   */
  var camelizeRE = /-(\w)/g;
  var camelize = cached(function (str) {
    return str.replace(camelizeRE, function (_, c) { return c ? c.toUpperCase() : ''; })
  });

  /**
   * Capitalize a string.
   */
  var capitalize = cached(function (str) {
    return str.charAt(0).toUpperCase() + str.slice(1)
  });

  /**
   * Hyphenate a camelCase string.
   */
  var hyphenateRE = /\B([A-Z])/g;
  var hyphenate = cached(function (str) {
    return str.replace(hyphenateRE, '-$1').toLowerCase()
  });

  /**
   * Simple bind polyfill for environments that do not support it,
   * e.g., PhantomJS 1.x. Technically, we don't need this anymore
   * since native bind is now performant enough in most browsers.
   * But removing it would mean breaking code that was able to run in
   * PhantomJS 1.x, so this must be kept for backward compatibility.
   */

  /* istanbul ignore next */
  function polyfillBind (fn, ctx) {
    function boundFn (a) {
      var l = arguments.length;
      return l
        ? l > 1
          ? fn.apply(ctx, arguments)
          : fn.call(ctx, a)
        : fn.call(ctx)
    }

    boundFn._length = fn.length;
    return boundFn
  }

  function nativeBind (fn, ctx) {
    return fn.bind(ctx)
  }

  var bind = Function.prototype.bind
    ? nativeBind
    : polyfillBind;

  /**
   * Convert an Array-like object to a real Array.
   */
  function toArray (list, start) {
    start = start || 0;
    var i = list.length - start;
    var ret = new Array(i);
    while (i--) {
      ret[i] = list[i + start];
    }
    return ret
  }

  /**
   * Mix properties into target object.
   */
  function extend (to, _from) {
    for (var key in _from) {
      to[key] = _from[key];
    }
    return to
  }

  /**
   * Merge an Array of Objects into a single Object.
   */
  function toObject (arr) {
    var res = {};
    for (var i = 0; i < arr.length; i++) {
      if (arr[i]) {
        extend(res, arr[i]);
      }
    }
    return res
  }

  /* eslint-disable no-unused-vars */

  /**
   * Perform no operation.
   * Stubbing args to make Flow happy without leaving useless transpiled code
   * with ...rest (https://flow.org/blog/2017/05/07/Strict-Function-Call-Arity/).
   */
  function noop (a, b, c) {}

  /**
   * Always return false.
   */
  var no = function (a, b, c) { return false; };

  /* eslint-enable no-unused-vars */

  /**
   * Return the same value.
   */
  var identity = function (_) { return _; };

  /**
   * Check if two values are loosely equal - that is,
   * if they are plain objects, do they have the same shape?
   */
  function looseEqual (a, b) {
    if (a === b) { return true }
    var isObjectA = isObject(a);
    var isObjectB = isObject(b);
    if (isObjectA && isObjectB) {
      try {
        var isArrayA = Array.isArray(a);
        var isArrayB = Array.isArray(b);
        if (isArrayA && isArrayB) {
          return a.length === b.length && a.every(function (e, i) {
            return looseEqual(e, b[i])
          })
        } else if (a instanceof Date && b instanceof Date) {
          return a.getTime() === b.getTime()
        } else if (!isArrayA && !isArrayB) {
          var keysA = Object.keys(a);
          var keysB = Object.keys(b);
          return keysA.length === keysB.length && keysA.every(function (key) {
            return looseEqual(a[key], b[key])
          })
        } else {
          /* istanbul ignore next */
          return false
        }
      } catch (e) {
        /* istanbul ignore next */
        return false
      }
    } else if (!isObjectA && !isObjectB) {
      return String(a) === String(b)
    } else {
      return false
    }
  }

  /**
   * Return the first index at which a loosely equal value can be
   * found in the array (if value is a plain object, the array must
   * contain an object of the same shape), or -1 if it is not present.
   */
  function looseIndexOf (arr, val) {
    for (var i = 0; i < arr.length; i++) {
      if (looseEqual(arr[i], val)) { return i }
    }
    return -1
  }

  /**
   * Ensure a function is called only once.
   */
  function once (fn) {
    var called = false;
    return function () {
      if (!called) {
        called = true;
        fn.apply(this, arguments);
      }
    }
  }

  var SSR_ATTR = 'data-server-rendered';

  var ASSET_TYPES = [
    'component',
    'directive',
    'filter'
  ];

  var LIFECYCLE_HOOKS = [
    'beforeCreate',
    'created',
    'beforeMount',
    'mounted',
    'beforeUpdate',
    'updated',
    'beforeDestroy',
    'destroyed',
    'activated',
    'deactivated',
    'errorCaptured',
    'serverPrefetch'
  ];

  /*  */



  var config = ({
    /**
     * Option merge strategies (used in core/util/options)
     */
    // $flow-disable-line
    optionMergeStrategies: Object.create(null),

    /**
     * Whether to suppress warnings.
     */
    silent: false,

    /**
     * Show production mode tip message on boot?
     */
    productionTip: "development" !== 'production',

    /**
     * Whether to enable devtools
     */
    devtools: "development" !== 'production',

    /**
     * Whether to record perf
     */
    performance: false,

    /**
     * Error handler for watcher errors
     */
    errorHandler: null,

    /**
     * Warn handler for watcher warns
     */
    warnHandler: null,

    /**
     * Ignore certain custom elements
     */
    ignoredElements: [],

    /**
     * Custom user key aliases for v-on
     */
    // $flow-disable-line
    keyCodes: Object.create(null),

    /**
     * Check if a tag is reserved so that it cannot be registered as a
     * component. This is platform-dependent and may be overwritten.
     */
    isReservedTag: no,

    /**
     * Check if an attribute is reserved so that it cannot be used as a component
     * prop. This is platform-dependent and may be overwritten.
     */
    isReservedAttr: no,

    /**
     * Check if a tag is an unknown element.
     * Platform-dependent.
     */
    isUnknownElement: no,

    /**
     * Get the namespace of an element
     */
    getTagNamespace: noop,

    /**
     * Parse the real tag name for the specific platform.
     */
    parsePlatformTagName: identity,

    /**
     * Check if an attribute must be bound using property, e.g. value
     * Platform-dependent.
     */
    mustUseProp: no,

    /**
     * Perform updates asynchronously. Intended to be used by Vue Test Utils
     * This will significantly reduce performance if set to false.
     */
    async: true,

    /**
     * Exposed for legacy reasons
     */
    _lifecycleHooks: LIFECYCLE_HOOKS
  });

  /*  */

  /**
   * unicode letters used for parsing html tags, component names and property paths.
   * using https://www.w3.org/TR/html53/semantics-scripting.html#potentialcustomelementname
   * skipping \u10000-\uEFFFF due to it freezing up PhantomJS
   */
  var unicodeRegExp = /a-zA-Z\u00B7\u00C0-\u00D6\u00D8-\u00F6\u00F8-\u037D\u037F-\u1FFF\u200C-\u200D\u203F-\u2040\u2070-\u218F\u2C00-\u2FEF\u3001-\uD7FF\uF900-\uFDCF\uFDF0-\uFFFD/;

  /**
   * Check if a string starts with $ or _
   */
  function isReserved (str) {
    var c = (str + '').charCodeAt(0);
    return c === 0x24 || c === 0x5F
  }

  /**
   * Define a property.
   */
  function def (obj, key, val, enumerable) {
    Object.defineProperty(obj, key, {
      value: val,
      enumerable: !!enumerable,
      writable: true,
      configurable: true
    });
  }

  /**
   * Parse simple path.
   */
  var bailRE = new RegExp(("[^" + (unicodeRegExp.source) + ".$_\\d]"));
  function parsePath (path) {
    if (bailRE.test(path)) {
      return
    }
    var segments = path.split('.');
    return function (obj) {
      for (var i = 0; i < segments.length; i++) {
        if (!obj) { return }
        obj = obj[segments[i]];
      }
      return obj
    }
  }

  /*  */

  // can we use __proto__?
  var hasProto = '__proto__' in {};

  // Browser environment sniffing
  var inBrowser = typeof window !== 'undefined';
  var inWeex = typeof WXEnvironment !== 'undefined' && !!WXEnvironment.platform;
  var weexPlatform = inWeex && WXEnvironment.platform.toLowerCase();
  var UA = inBrowser && window.navigator.userAgent.toLowerCase();
  var isIE = UA && /msie|trident/.test(UA);
  var isIE9 = UA && UA.indexOf('msie 9.0') > 0;
  var isEdge = UA && UA.indexOf('edge/') > 0;
  var isAndroid = (UA && UA.indexOf('android') > 0) || (weexPlatform === 'android');
  var isIOS = (UA && /iphone|ipad|ipod|ios/.test(UA)) || (weexPlatform === 'ios');
  var isChrome = UA && /chrome\/\d+/.test(UA) && !isEdge;
  var isPhantomJS = UA && /phantomjs/.test(UA);
  var isFF = UA && UA.match(/firefox\/(\d+)/);

  // Firefox has a "watch" function on Object.prototype...
  var nativeWatch = ({}).watch;

  var supportsPassive = false;
  if (inBrowser) {
    try {
      var opts = {};
      Object.defineProperty(opts, 'passive', ({
        get: function get () {
          /* istanbul ignore next */
          supportsPassive = true;
        }
      })); // https://github.com/facebook/flow/issues/285
      window.addEventListener('test-passive', null, opts);
    } catch (e) {}
  }

  // this needs to be lazy-evaled because vue may be required before
  // vue-server-renderer can set VUE_ENV
  var _isServer;
  var isServerRendering = function () {
    if (_isServer === undefined) {
      /* istanbul ignore if */
      if (!inBrowser && !inWeex && typeof global !== 'undefined') {
        // detect presence of vue-server-renderer and avoid
        // Webpack shimming the process
        _isServer = global['process'] && global['process'].env.VUE_ENV === 'server';
      } else {
        _isServer = false;
      }
    }
    return _isServer
  };

  // detect devtools
  var devtools = inBrowser && window.__VUE_DEVTOOLS_GLOBAL_HOOK__;

  /* istanbul ignore next */
  function isNative (Ctor) {
    return typeof Ctor === 'function' && /native code/.test(Ctor.toString())
  }

  var hasSymbol =
    typeof Symbol !== 'undefined' && isNative(Symbol) &&
    typeof Reflect !== 'undefined' && isNative(Reflect.ownKeys);

  var _Set;
  /* istanbul ignore if */ // $flow-disable-line
  if (typeof Set !== 'undefined' && isNative(Set)) {
    // use native Set when available.
    _Set = Set;
  } else {
    // a non-standard Set polyfill that only works with primitive keys.
    _Set = /*@__PURE__*/(function () {
      function Set () {
        this.set = Object.create(null);
      }
      Set.prototype.has = function has (key) {
        return this.set[key] === true
      };
      Set.prototype.add = function add (key) {
        this.set[key] = true;
      };
      Set.prototype.clear = function clear () {
        this.set = Object.create(null);
      };

      return Set;
    }());
  }

  /*  */

  var warn = noop;
  var tip = noop;
  var generateComponentTrace = (noop); // work around flow check
  var formatComponentName = (noop);

  {
    var hasConsole = typeof console !== 'undefined';
    var classifyRE = /(?:^|[-_])(\w)/g;
    var classify = function (str) { return str
      .replace(classifyRE, function (c) { return c.toUpperCase(); })
      .replace(/[-_]/g, ''); };

    warn = function (msg, vm) {
      var trace = vm ? generateComponentTrace(vm) : '';

      if (config.warnHandler) {
        config.warnHandler.call(null, msg, vm, trace);
      } else if (hasConsole && (!config.silent)) {
        console.error(("[Vue warn]: " + msg + trace));
      }
    };

    tip = function (msg, vm) {
      if (hasConsole && (!config.silent)) {
        console.warn("[Vue tip]: " + msg + (
          vm ? generateComponentTrace(vm) : ''
        ));
      }
    };

    formatComponentName = function (vm, includeFile) {
      if (vm.$root === vm) {
        return '<Root>'
      }
      var options = typeof vm === 'function' && vm.cid != null
        ? vm.options
        : vm._isVue
          ? vm.$options || vm.constructor.options
          : vm;
      var name = options.name || options._componentTag;
      var file = options.__file;
      if (!name && file) {
        var match = file.match(/([^/\\]+)\.vue$/);
        name = match && match[1];
      }

      return (
        (name ? ("<" + (classify(name)) + ">") : "<Anonymous>") +
        (file && includeFile !== false ? (" at " + file) : '')
      )
    };

    var repeat = function (str, n) {
      var res = '';
      while (n) {
        if (n % 2 === 1) { res += str; }
        if (n > 1) { str += str; }
        n >>= 1;
      }
      return res
    };

    generateComponentTrace = function (vm) {
      if (vm._isVue && vm.$parent) {
        var tree = [];
        var currentRecursiveSequence = 0;
        while (vm) {
          if (tree.length > 0) {
            var last = tree[tree.length - 1];
            if (last.constructor === vm.constructor) {
              currentRecursiveSequence++;
              vm = vm.$parent;
              continue
            } else if (currentRecursiveSequence > 0) {
              tree[tree.length - 1] = [last, currentRecursiveSequence];
              currentRecursiveSequence = 0;
            }
          }
          tree.push(vm);
          vm = vm.$parent;
        }
        return '\n\nfound in\n\n' + tree
          .map(function (vm, i) { return ("" + (i === 0 ? '---> ' : repeat(' ', 5 + i * 2)) + (Array.isArray(vm)
              ? ((formatComponentName(vm[0])) + "... (" + (vm[1]) + " recursive calls)")
              : formatComponentName(vm))); })
          .join('\n')
      } else {
        return ("\n\n(found in " + (formatComponentName(vm)) + ")")
      }
    };
  }

  /*  */

  var uid = 0;

  /**
   * A dep is an observable that can have multiple
   * directives subscribing to it.
   */
  var Dep = function Dep () {
    this.id = uid++;
    this.subs = [];
  };

  Dep.prototype.addSub = function addSub (sub) {
    this.subs.push(sub);
  };

  Dep.prototype.removeSub = function removeSub (sub) {
    remove(this.subs, sub);
  };

  Dep.prototype.depend = function depend () {
    if (Dep.target) {
      Dep.target.addDep(this);
    }
  };

  Dep.prototype.notify = function notify () {
    // stabilize the subscriber list first
    var subs = this.subs.slice();
    if (!config.async) {
      // subs aren't sorted in scheduler if not running async
      // we need to sort them now to make sure they fire in correct
      // order
      subs.sort(function (a, b) { return a.id - b.id; });
    }
    for (var i = 0, l = subs.length; i < l; i++) {
      subs[i].update();
    }
  };

  // The current target watcher being evaluated.
  // This is globally unique because only one watcher
  // can be evaluated at a time.
  Dep.target = null;
  var targetStack = [];

  function pushTarget (target) {
    targetStack.push(target);
    Dep.target = target;
  }

  function popTarget () {
    targetStack.pop();
    Dep.target = targetStack[targetStack.length - 1];
  }

  /*  */

  var VNode = function VNode (
    tag,
    data,
    children,
    text,
    elm,
    context,
    componentOptions,
    asyncFactory
  ) {
    this.tag = tag;
    this.data = data;
    this.children = children;
    this.text = text;
    this.elm = elm;
    this.ns = undefined;
    this.context = context;
    this.fnContext = undefined;
    this.fnOptions = undefined;
    this.fnScopeId = undefined;
    this.key = data && data.key;
    this.componentOptions = componentOptions;
    this.componentInstance = undefined;
    this.parent = undefined;
    this.raw = false;
    this.isStatic = false;
    this.isRootInsert = true;
    this.isComment = false;
    this.isCloned = false;
    this.isOnce = false;
    this.asyncFactory = asyncFactory;
    this.asyncMeta = undefined;
    this.isAsyncPlaceholder = false;
  };

  var prototypeAccessors = { child: { configurable: true } };

  // DEPRECATED: alias for componentInstance for backwards compat.
  /* istanbul ignore next */
  prototypeAccessors.child.get = function () {
    return this.componentInstance
  };

  Object.defineProperties( VNode.prototype, prototypeAccessors );

  var createEmptyVNode = function (text) {
    if ( text === void 0 ) text = '';

    var node = new VNode();
    node.text = text;
    node.isComment = true;
    return node
  };

  function createTextVNode (val) {
    return new VNode(undefined, undefined, undefined, String(val))
  }

  // optimized shallow clone
  // used for static nodes and slot nodes because they may be reused across
  // multiple renders, cloning them avoids errors when DOM manipulations rely
  // on their elm reference.
  function cloneVNode (vnode) {
    var cloned = new VNode(
      vnode.tag,
      vnode.data,
      // #7975
      // clone children array to avoid mutating original in case of cloning
      // a child.
      vnode.children && vnode.children.slice(),
      vnode.text,
      vnode.elm,
      vnode.context,
      vnode.componentOptions,
      vnode.asyncFactory
    );
    cloned.ns = vnode.ns;
    cloned.isStatic = vnode.isStatic;
    cloned.key = vnode.key;
    cloned.isComment = vnode.isComment;
    cloned.fnContext = vnode.fnContext;
    cloned.fnOptions = vnode.fnOptions;
    cloned.fnScopeId = vnode.fnScopeId;
    cloned.asyncMeta = vnode.asyncMeta;
    cloned.isCloned = true;
    return cloned
  }

  /*
   * not type checking this file because flow doesn't play well with
   * dynamically accessing methods on Array prototype
   */

  var arrayProto = Array.prototype;
  var arrayMethods = Object.create(arrayProto);

  var methodsToPatch = [
    'push',
    'pop',
    'shift',
    'unshift',
    'splice',
    'sort',
    'reverse'
  ];

  /**
   * Intercept mutating methods and emit events
   */
  methodsToPatch.forEach(function (method) {
    // cache original method
    var original = arrayProto[method];
    def(arrayMethods, method, function mutator () {
      var args = [], len = arguments.length;
      while ( len-- ) args[ len ] = arguments[ len ];

      var result = original.apply(this, args);
      var ob = this.__ob__;
      var inserted;
      switch (method) {
        case 'push':
        case 'unshift':
          inserted = args;
          break
        case 'splice':
          inserted = args.slice(2);
          break
      }
      if (inserted) { ob.observeArray(inserted); }
      // notify change
      ob.dep.notify();
      return result
    });
  });

  /*  */

  var arrayKeys = Object.getOwnPropertyNames(arrayMethods);

  /**
   * In some cases we may want to disable observation inside a component's
   * update computation.
   */
  var shouldObserve = true;

  function toggleObserving (value) {
    shouldObserve = value;
  }

  /**
   * Observer class that is attached to each observed
   * object. Once attached, the observer converts the target
   * object's property keys into getter/setters that
   * collect dependencies and dispatch updates.
   */
  var Observer = function Observer (value) {
    this.value = value;
    this.dep = new Dep();
    this.vmCount = 0;
    def(value, '__ob__', this);
    if (Array.isArray(value)) {
      if (hasProto) {
        protoAugment(value, arrayMethods);
      } else {
        copyAugment(value, arrayMethods, arrayKeys);
      }
      this.observeArray(value);
    } else {
      this.walk(value);
    }
  };

  /**
   * Walk through all properties and convert them into
   * getter/setters. This method should only be called when
   * value type is Object.
   */
  Observer.prototype.walk = function walk (obj) {
    var keys = Object.keys(obj);
    for (var i = 0; i < keys.length; i++) {
      defineReactive$$1(obj, keys[i]);
    }
  };

  /**
   * Observe a list of Array items.
   */
  Observer.prototype.observeArray = function observeArray (items) {
    for (var i = 0, l = items.length; i < l; i++) {
      observe(items[i]);
    }
  };

  // helpers

  /**
   * Augment a target Object or Array by intercepting
   * the prototype chain using __proto__
   */
  function protoAugment (target, src) {
    /* eslint-disable no-proto */
    target.__proto__ = src;
    /* eslint-enable no-proto */
  }

  /**
   * Augment a target Object or Array by defining
   * hidden properties.
   */
  /* istanbul ignore next */
  function copyAugment (target, src, keys) {
    for (var i = 0, l = keys.length; i < l; i++) {
      var key = keys[i];
      def(target, key, src[key]);
    }
  }

  /**
   * Attempt to create an observer instance for a value,
   * returns the new observer if successfully observed,
   * or the existing observer if the value already has one.
   */
  function observe (value, asRootData) {
    if (!isObject(value) || value instanceof VNode) {
      return
    }
    var ob;
    if (hasOwn(value, '__ob__') && value.__ob__ instanceof Observer) {
      ob = value.__ob__;
    } else if (
      shouldObserve &&
      !isServerRendering() &&
      (Array.isArray(value) || isPlainObject(value)) &&
      Object.isExtensible(value) &&
      !value._isVue
    ) {
      ob = new Observer(value);
    }
    if (asRootData && ob) {
      ob.vmCount++;
    }
    return ob
  }

  /**
   * Define a reactive property on an Object.
   */
  function defineReactive$$1 (
    obj,
    key,
    val,
    customSetter,
    shallow
  ) {
    var dep = new Dep();

    var property = Object.getOwnPropertyDescriptor(obj, key);
    if (property && property.configurable === false) {
      return
    }

    // cater for pre-defined getter/setters
    var getter = property && property.get;
    var setter = property && property.set;
    if ((!getter || setter) && arguments.length === 2) {
      val = obj[key];
    }

    var childOb = !shallow && observe(val);
    Object.defineProperty(obj, key, {
      enumerable: true,
      configurable: true,
      get: function reactiveGetter () {
        var value = getter ? getter.call(obj) : val;
        if (Dep.target) {
          dep.depend();
          if (childOb) {
            childOb.dep.depend();
            if (Array.isArray(value)) {
              dependArray(value);
            }
          }
        }
        return value
      },
      set: function reactiveSetter (newVal) {
        var value = getter ? getter.call(obj) : val;
        /* eslint-disable no-self-compare */
        if (newVal === value || (newVal !== newVal && value !== value)) {
          return
        }
        /* eslint-enable no-self-compare */
        if (customSetter) {
          customSetter();
        }
        // #7981: for accessor properties without setter
        if (getter && !setter) { return }
        if (setter) {
          setter.call(obj, newVal);
        } else {
          val = newVal;
        }
        childOb = !shallow && observe(newVal);
        dep.notify();
      }
    });
  }

  /**
   * Set a property on an object. Adds the new property and
   * triggers change notification if the property doesn't
   * already exist.
   */
  function set (target, key, val) {
    if (isUndef(target) || isPrimitive(target)
    ) {
      warn(("Cannot set reactive property on undefined, null, or primitive value: " + ((target))));
    }
    if (Array.isArray(target) && isValidArrayIndex(key)) {
      target.length = Math.max(target.length, key);
      target.splice(key, 1, val);
      return val
    }
    if (key in target && !(key in Object.prototype)) {
      target[key] = val;
      return val
    }
    var ob = (target).__ob__;
    if (target._isVue || (ob && ob.vmCount)) {
      warn(
        'Avoid adding reactive properties to a Vue instance or its root $data ' +
        'at runtime - declare it upfront in the data option.'
      );
      return val
    }
    if (!ob) {
      target[key] = val;
      return val
    }
    defineReactive$$1(ob.value, key, val);
    ob.dep.notify();
    return val
  }

  /**
   * Delete a property and trigger change if necessary.
   */
  function del (target, key) {
    if (isUndef(target) || isPrimitive(target)
    ) {
      warn(("Cannot delete reactive property on undefined, null, or primitive value: " + ((target))));
    }
    if (Array.isArray(target) && isValidArrayIndex(key)) {
      target.splice(key, 1);
      return
    }
    var ob = (target).__ob__;
    if (target._isVue || (ob && ob.vmCount)) {
      warn(
        'Avoid deleting properties on a Vue instance or its root $data ' +
        '- just set it to null.'
      );
      return
    }
    if (!hasOwn(target, key)) {
      return
    }
    delete target[key];
    if (!ob) {
      return
    }
    ob.dep.notify();
  }

  /**
   * Collect dependencies on array elements when the array is touched, since
   * we cannot intercept array element access like property getters.
   */
  function dependArray (value) {
    for (var e = (void 0), i = 0, l = value.length; i < l; i++) {
      e = value[i];
      e && e.__ob__ && e.__ob__.dep.depend();
      if (Array.isArray(e)) {
        dependArray(e);
      }
    }
  }

  /*  */

  /**
   * Option overwriting strategies are functions that handle
   * how to merge a parent option value and a child option
   * value into the final value.
   */
  var strats = config.optionMergeStrategies;

  /**
   * Options with restrictions
   */
  {
    strats.el = strats.propsData = function (parent, child, vm, key) {
      if (!vm) {
        warn(
          "option \"" + key + "\" can only be used during instance " +
          'creation with the `new` keyword.'
        );
      }
      return defaultStrat(parent, child)
    };
  }

  /**
   * Helper that recursively merges two data objects together.
   */
  function mergeData (to, from) {
    if (!from) { return to }
    var key, toVal, fromVal;

    var keys = hasSymbol
      ? Reflect.ownKeys(from)
      : Object.keys(from);

    for (var i = 0; i < keys.length; i++) {
      key = keys[i];
      // in case the object is already observed...
      if (key === '__ob__') { continue }
      toVal = to[key];
      fromVal = from[key];
      if (!hasOwn(to, key)) {
        set(to, key, fromVal);
      } else if (
        toVal !== fromVal &&
        isPlainObject(toVal) &&
        isPlainObject(fromVal)
      ) {
        mergeData(toVal, fromVal);
      }
    }
    return to
  }

  /**
   * Data
   */
  function mergeDataOrFn (
    parentVal,
    childVal,
    vm
  ) {
    if (!vm) {
      // in a Vue.extend merge, both should be functions
      if (!childVal) {
        return parentVal
      }
      if (!parentVal) {
        return childVal
      }
      // when parentVal & childVal are both present,
      // we need to return a function that returns the
      // merged result of both functions... no need to
      // check if parentVal is a function here because
      // it has to be a function to pass previous merges.
      return function mergedDataFn () {
        return mergeData(
          typeof childVal === 'function' ? childVal.call(this, this) : childVal,
          typeof parentVal === 'function' ? parentVal.call(this, this) : parentVal
        )
      }
    } else {
      return function mergedInstanceDataFn () {
        // instance merge
        var instanceData = typeof childVal === 'function'
          ? childVal.call(vm, vm)
          : childVal;
        var defaultData = typeof parentVal === 'function'
          ? parentVal.call(vm, vm)
          : parentVal;
        if (instanceData) {
          return mergeData(instanceData, defaultData)
        } else {
          return defaultData
        }
      }
    }
  }

  strats.data = function (
    parentVal,
    childVal,
    vm
  ) {
    if (!vm) {
      if (childVal && typeof childVal !== 'function') {
        warn(
          'The "data" option should be a function ' +
          'that returns a per-instance value in component ' +
          'definitions.',
          vm
        );

        return parentVal
      }
      return mergeDataOrFn(parentVal, childVal)
    }

    return mergeDataOrFn(parentVal, childVal, vm)
  };

  /**
   * Hooks and props are merged as arrays.
   */
  function mergeHook (
    parentVal,
    childVal
  ) {
    var res = childVal
      ? parentVal
        ? parentVal.concat(childVal)
        : Array.isArray(childVal)
          ? childVal
          : [childVal]
      : parentVal;
    return res
      ? dedupeHooks(res)
      : res
  }

  function dedupeHooks (hooks) {
    var res = [];
    for (var i = 0; i < hooks.length; i++) {
      if (res.indexOf(hooks[i]) === -1) {
        res.push(hooks[i]);
      }
    }
    return res
  }

  LIFECYCLE_HOOKS.forEach(function (hook) {
    strats[hook] = mergeHook;
  });

  /**
   * Assets
   *
   * When a vm is present (instance creation), we need to do
   * a three-way merge between constructor options, instance
   * options and parent options.
   */
  function mergeAssets (
    parentVal,
    childVal,
    vm,
    key
  ) {
    var res = Object.create(parentVal || null);
    if (childVal) {
      assertObjectType(key, childVal, vm);
      return extend(res, childVal)
    } else {
      return res
    }
  }

  ASSET_TYPES.forEach(function (type) {
    strats[type + 's'] = mergeAssets;
  });

  /**
   * Watchers.
   *
   * Watchers hashes should not overwrite one
   * another, so we merge them as arrays.
   */
  strats.watch = function (
    parentVal,
    childVal,
    vm,
    key
  ) {
    // work around Firefox's Object.prototype.watch...
    if (parentVal === nativeWatch) { parentVal = undefined; }
    if (childVal === nativeWatch) { childVal = undefined; }
    /* istanbul ignore if */
    if (!childVal) { return Object.create(parentVal || null) }
    {
      assertObjectType(key, childVal, vm);
    }
    if (!parentVal) { return childVal }
    var ret = {};
    extend(ret, parentVal);
    for (var key$1 in childVal) {
      var parent = ret[key$1];
      var child = childVal[key$1];
      if (parent && !Array.isArray(parent)) {
        parent = [parent];
      }
      ret[key$1] = parent
        ? parent.concat(child)
        : Array.isArray(child) ? child : [child];
    }
    return ret
  };

  /**
   * Other object hashes.
   */
  strats.props =
  strats.methods =
  strats.inject =
  strats.computed = function (
    parentVal,
    childVal,
    vm,
    key
  ) {
    if (childVal && "development" !== 'production') {
      assertObjectType(key, childVal, vm);
    }
    if (!parentVal) { return childVal }
    var ret = Object.create(null);
    extend(ret, parentVal);
    if (childVal) { extend(ret, childVal); }
    return ret
  };
  strats.provide = mergeDataOrFn;

  /**
   * Default strategy.
   */
  var defaultStrat = function (parentVal, childVal) {
    return childVal === undefined
      ? parentVal
      : childVal
  };

  /**
   * Validate component names
   */
  function checkComponents (options) {
    for (var key in options.components) {
      validateComponentName(key);
    }
  }

  function validateComponentName (name) {
    if (!new RegExp(("^[a-zA-Z][\\-\\.0-9_" + (unicodeRegExp.source) + "]*$")).test(name)) {
      warn(
        'Invalid component name: "' + name + '". Component names ' +
        'should conform to valid custom element name in html5 specification.'
      );
    }
    if (isBuiltInTag(name) || config.isReservedTag(name)) {
      warn(
        'Do not use built-in or reserved HTML elements as component ' +
        'id: ' + name
      );
    }
  }

  /**
   * Ensure all props option syntax are normalized into the
   * Object-based format.
   */
  function normalizeProps (options, vm) {
    var props = options.props;
    if (!props) { return }
    var res = {};
    var i, val, name;
    if (Array.isArray(props)) {
      i = props.length;
      while (i--) {
        val = props[i];
        if (typeof val === 'string') {
          name = camelize(val);
          res[name] = { type: null };
        } else {
          warn('props must be strings when using array syntax.');
        }
      }
    } else if (isPlainObject(props)) {
      for (var key in props) {
        val = props[key];
        name = camelize(key);
        res[name] = isPlainObject(val)
          ? val
          : { type: val };
      }
    } else {
      warn(
        "Invalid value for option \"props\": expected an Array or an Object, " +
        "but got " + (toRawType(props)) + ".",
        vm
      );
    }
    options.props = res;
  }

  /**
   * Normalize all injections into Object-based format
   */
  function normalizeInject (options, vm) {
    var inject = options.inject;
    if (!inject) { return }
    var normalized = options.inject = {};
    if (Array.isArray(inject)) {
      for (var i = 0; i < inject.length; i++) {
        normalized[inject[i]] = { from: inject[i] };
      }
    } else if (isPlainObject(inject)) {
      for (var key in inject) {
        var val = inject[key];
        normalized[key] = isPlainObject(val)
          ? extend({ from: key }, val)
          : { from: val };
      }
    } else {
      warn(
        "Invalid value for option \"inject\": expected an Array or an Object, " +
        "but got " + (toRawType(inject)) + ".",
        vm
      );
    }
  }

  /**
   * Normalize raw function directives into object format.
   */
  function normalizeDirectives (options) {
    var dirs = options.directives;
    if (dirs) {
      for (var key in dirs) {
        var def$$1 = dirs[key];
        if (typeof def$$1 === 'function') {
          dirs[key] = { bind: def$$1, update: def$$1 };
        }
      }
    }
  }

  function assertObjectType (name, value, vm) {
    if (!isPlainObject(value)) {
      warn(
        "Invalid value for option \"" + name + "\": expected an Object, " +
        "but got " + (toRawType(value)) + ".",
        vm
      );
    }
  }

  /**
   * Merge two option objects into a new one.
   * Core utility used in both instantiation and inheritance.
   */
  function mergeOptions (
    parent,
    child,
    vm
  ) {
    {
      checkComponents(child);
    }

    if (typeof child === 'function') {
      child = child.options;
    }

    normalizeProps(child, vm);
    normalizeInject(child, vm);
    normalizeDirectives(child);

    // Apply extends and mixins on the child options,
    // but only if it is a raw options object that isn't
    // the result of another mergeOptions call.
    // Only merged options has the _base property.
    if (!child._base) {
      if (child.extends) {
        parent = mergeOptions(parent, child.extends, vm);
      }
      if (child.mixins) {
        for (var i = 0, l = child.mixins.length; i < l; i++) {
          parent = mergeOptions(parent, child.mixins[i], vm);
        }
      }
    }

    var options = {};
    var key;
    for (key in parent) {
      mergeField(key);
    }
    for (key in child) {
      if (!hasOwn(parent, key)) {
        mergeField(key);
      }
    }
    function mergeField (key) {
      var strat = strats[key] || defaultStrat;
      options[key] = strat(parent[key], child[key], vm, key);
    }
    return options
  }

  /**
   * Resolve an asset.
   * This function is used because child instances need access
   * to assets defined in its ancestor chain.
   */
  function resolveAsset (
    options,
    type,
    id,
    warnMissing
  ) {
    /* istanbul ignore if */
    if (typeof id !== 'string') {
      return
    }
    var assets = options[type];
    // check local registration variations first
    if (hasOwn(assets, id)) { return assets[id] }
    var camelizedId = camelize(id);
    if (hasOwn(assets, camelizedId)) { return assets[camelizedId] }
    var PascalCaseId = capitalize(camelizedId);
    if (hasOwn(assets, PascalCaseId)) { return assets[PascalCaseId] }
    // fallback to prototype chain
    var res = assets[id] || assets[camelizedId] || assets[PascalCaseId];
    if (warnMissing && !res) {
      warn(
        'Failed to resolve ' + type.slice(0, -1) + ': ' + id,
        options
      );
    }
    return res
  }

  /*  */



  function validateProp (
    key,
    propOptions,
    propsData,
    vm
  ) {
    var prop = propOptions[key];
    var absent = !hasOwn(propsData, key);
    var value = propsData[key];
    // boolean casting
    var booleanIndex = getTypeIndex(Boolean, prop.type);
    if (booleanIndex > -1) {
      if (absent && !hasOwn(prop, 'default')) {
        value = false;
      } else if (value === '' || value === hyphenate(key)) {
        // only cast empty string / same name to boolean if
        // boolean has higher priority
        var stringIndex = getTypeIndex(String, prop.type);
        if (stringIndex < 0 || booleanIndex < stringIndex) {
          value = true;
        }
      }
    }
    // check default value
    if (value === undefined) {
      value = getPropDefaultValue(vm, prop, key);
      // since the default value is a fresh copy,
      // make sure to observe it.
      var prevShouldObserve = shouldObserve;
      toggleObserving(true);
      observe(value);
      toggleObserving(prevShouldObserve);
    }
    {
      assertProp(prop, key, value, vm, absent);
    }
    return value
  }

  /**
   * Get the default value of a prop.
   */
  function getPropDefaultValue (vm, prop, key) {
    // no default, return undefined
    if (!hasOwn(prop, 'default')) {
      return undefined
    }
    var def = prop.default;
    // warn against non-factory defaults for Object & Array
    if (isObject(def)) {
      warn(
        'Invalid default value for prop "' + key + '": ' +
        'Props with type Object/Array must use a factory function ' +
        'to return the default value.',
        vm
      );
    }
    // the raw prop value was also undefined from previous render,
    // return previous default value to avoid unnecessary watcher trigger
    if (vm && vm.$options.propsData &&
      vm.$options.propsData[key] === undefined &&
      vm._props[key] !== undefined
    ) {
      return vm._props[key]
    }
    // call factory function for non-Function types
    // a value is Function if its prototype is function even across different execution context
    return typeof def === 'function' && getType(prop.type) !== 'Function'
      ? def.call(vm)
      : def
  }

  /**
   * Assert whether a prop is valid.
   */
  function assertProp (
    prop,
    name,
    value,
    vm,
    absent
  ) {
    if (prop.required && absent) {
      warn(
        'Missing required prop: "' + name + '"',
        vm
      );
      return
    }
    if (value == null && !prop.required) {
      return
    }
    var type = prop.type;
    var valid = !type || type === true;
    var expectedTypes = [];
    if (type) {
      if (!Array.isArray(type)) {
        type = [type];
      }
      for (var i = 0; i < type.length && !valid; i++) {
        var assertedType = assertType(value, type[i]);
        expectedTypes.push(assertedType.expectedType || '');
        valid = assertedType.valid;
      }
    }

    if (!valid) {
      warn(
        getInvalidTypeMessage(name, value, expectedTypes),
        vm
      );
      return
    }
    var validator = prop.validator;
    if (validator) {
      if (!validator(value)) {
        warn(
          'Invalid prop: custom validator check failed for prop "' + name + '".',
          vm
        );
      }
    }
  }

  var simpleCheckRE = /^(String|Number|Boolean|Function|Symbol)$/;

  function assertType (value, type) {
    var valid;
    var expectedType = getType(type);
    if (simpleCheckRE.test(expectedType)) {
      var t = typeof value;
      valid = t === expectedType.toLowerCase();
      // for primitive wrapper objects
      if (!valid && t === 'object') {
        valid = value instanceof type;
      }
    } else if (expectedType === 'Object') {
      valid = isPlainObject(value);
    } else if (expectedType === 'Array') {
      valid = Array.isArray(value);
    } else {
      valid = value instanceof type;
    }
    return {
      valid: valid,
      expectedType: expectedType
    }
  }

  /**
   * Use function string name to check built-in types,
   * because a simple equality check will fail when running
   * across different vms / iframes.
   */
  function getType (fn) {
    var match = fn && fn.toString().match(/^\s*function (\w+)/);
    return match ? match[1] : ''
  }

  function isSameType (a, b) {
    return getType(a) === getType(b)
  }

  function getTypeIndex (type, expectedTypes) {
    if (!Array.isArray(expectedTypes)) {
      return isSameType(expectedTypes, type) ? 0 : -1
    }
    for (var i = 0, len = expectedTypes.length; i < len; i++) {
      if (isSameType(expectedTypes[i], type)) {
        return i
      }
    }
    return -1
  }

  function getInvalidTypeMessage (name, value, expectedTypes) {
    var message = "Invalid prop: type check failed for prop \"" + name + "\"." +
      " Expected " + (expectedTypes.map(capitalize).join(', '));
    var expectedType = expectedTypes[0];
    var receivedType = toRawType(value);
    var expectedValue = styleValue(value, expectedType);
    var receivedValue = styleValue(value, receivedType);
    // check if we need to specify expected value
    if (expectedTypes.length === 1 &&
        isExplicable(expectedType) &&
        !isBoolean(expectedType, receivedType)) {
      message += " with value " + expectedValue;
    }
    message += ", got " + receivedType + " ";
    // check if we need to specify received value
    if (isExplicable(receivedType)) {
      message += "with value " + receivedValue + ".";
    }
    return message
  }

  function styleValue (value, type) {
    if (type === 'String') {
      return ("\"" + value + "\"")
    } else if (type === 'Number') {
      return ("" + (Number(value)))
    } else {
      return ("" + value)
    }
  }

  function isExplicable (value) {
    var explicitTypes = ['string', 'number', 'boolean'];
    return explicitTypes.some(function (elem) { return value.toLowerCase() === elem; })
  }

  function isBoolean () {
    var args = [], len = arguments.length;
    while ( len-- ) args[ len ] = arguments[ len ];

    return args.some(function (elem) { return elem.toLowerCase() === 'boolean'; })
  }

  /*  */

  function handleError (err, vm, info) {
    // Deactivate deps tracking while processing error handler to avoid possible infinite rendering.
    // See: https://github.com/vuejs/vuex/issues/1505
    pushTarget();
    try {
      if (vm) {
        var cur = vm;
        while ((cur = cur.$parent)) {
          var hooks = cur.$options.errorCaptured;
          if (hooks) {
            for (var i = 0; i < hooks.length; i++) {
              try {
                var capture = hooks[i].call(cur, err, vm, info) === false;
                if (capture) { return }
              } catch (e) {
                globalHandleError(e, cur, 'errorCaptured hook');
              }
            }
          }
        }
      }
      globalHandleError(err, vm, info);
    } finally {
      popTarget();
    }
  }

  function invokeWithErrorHandling (
    handler,
    context,
    args,
    vm,
    info
  ) {
    var res;
    try {
      res = args ? handler.apply(context, args) : handler.call(context);
      if (res && !res._isVue && isPromise(res) && !res._handled) {
        res.catch(function (e) { return handleError(e, vm, info + " (Promise/async)"); });
        // issue #9511
        // avoid catch triggering multiple times when nested calls
        res._handled = true;
      }
    } catch (e) {
      handleError(e, vm, info);
    }
    return res
  }

  function globalHandleError (err, vm, info) {
    if (config.errorHandler) {
      try {
        return config.errorHandler.call(null, err, vm, info)
      } catch (e) {
        // if the user intentionally throws the original error in the handler,
        // do not log it twice
        if (e !== err) {
          logError(e, null, 'config.errorHandler');
        }
      }
    }
    logError(err, vm, info);
  }

  function logError (err, vm, info) {
    {
      warn(("Error in " + info + ": \"" + (err.toString()) + "\""), vm);
    }
    /* istanbul ignore else */
    if ((inBrowser || inWeex) && typeof console !== 'undefined') {
      console.error(err);
    } else {
      throw err
    }
  }

  /*  */

  var isUsingMicroTask = false;

  var callbacks = [];
  var pending = false;

  function flushCallbacks () {
    pending = false;
    var copies = callbacks.slice(0);
    callbacks.length = 0;
    for (var i = 0; i < copies.length; i++) {
      copies[i]();
    }
  }

  // Here we have async deferring wrappers using microtasks.
  // In 2.5 we used (macro) tasks (in combination with microtasks).
  // However, it has subtle problems when state is changed right before repaint
  // (e.g. #6813, out-in transitions).
  // Also, using (macro) tasks in event handler would cause some weird behaviors
  // that cannot be circumvented (e.g. #7109, #7153, #7546, #7834, #8109).
  // So we now use microtasks everywhere, again.
  // A major drawback of this tradeoff is that there are some scenarios
  // where microtasks have too high a priority and fire in between supposedly
  // sequential events (e.g. #4521, #6690, which have workarounds)
  // or even between bubbling of the same event (#6566).
  var timerFunc;

  // The nextTick behavior leverages the microtask queue, which can be accessed
  // via either native Promise.then or MutationObserver.
  // MutationObserver has wider support, however it is seriously bugged in
  // UIWebView in iOS >= 9.3.3 when triggered in touch event handlers. It
  // completely stops working after triggering a few times... so, if native
  // Promise is available, we will use it:
  /* istanbul ignore next, $flow-disable-line */
  if (typeof Promise !== 'undefined' && isNative(Promise)) {
    var p = Promise.resolve();
    timerFunc = function () {
      p.then(flushCallbacks);
      // In problematic UIWebViews, Promise.then doesn't completely break, but
      // it can get stuck in a weird state where callbacks are pushed into the
      // microtask queue but the queue isn't being flushed, until the browser
      // needs to do some other work, e.g. handle a timer. Therefore we can
      // "force" the microtask queue to be flushed by adding an empty timer.
      if (isIOS) { setTimeout(noop); }
    };
    isUsingMicroTask = true;
  } else if (!isIE && typeof MutationObserver !== 'undefined' && (
    isNative(MutationObserver) ||
    // PhantomJS and iOS 7.x
    MutationObserver.toString() === '[object MutationObserverConstructor]'
  )) {
    // Use MutationObserver where native Promise is not available,
    // e.g. PhantomJS, iOS7, Android 4.4
    // (#6466 MutationObserver is unreliable in IE11)
    var counter = 1;
    var observer = new MutationObserver(flushCallbacks);
    var textNode = document.createTextNode(String(counter));
    observer.observe(textNode, {
      characterData: true
    });
    timerFunc = function () {
      counter = (counter + 1) % 2;
      textNode.data = String(counter);
    };
    isUsingMicroTask = true;
  } else if (typeof setImmediate !== 'undefined' && isNative(setImmediate)) {
    // Fallback to setImmediate.
    // Techinically it leverages the (macro) task queue,
    // but it is still a better choice than setTimeout.
    timerFunc = function () {
      setImmediate(flushCallbacks);
    };
  } else {
    // Fallback to setTimeout.
    timerFunc = function () {
      setTimeout(flushCallbacks, 0);
    };
  }

  function nextTick (cb, ctx) {
    var _resolve;
    callbacks.push(function () {
      if (cb) {
        try {
          cb.call(ctx);
        } catch (e) {
          handleError(e, ctx, 'nextTick');
        }
      } else if (_resolve) {
        _resolve(ctx);
      }
    });
    if (!pending) {
      pending = true;
      timerFunc();
    }
    // $flow-disable-line
    if (!cb && typeof Promise !== 'undefined') {
      return new Promise(function (resolve) {
        _resolve = resolve;
      })
    }
  }

  /*  */

  /* not type checking this file because flow doesn't play well with Proxy */

  var initProxy;

  {
    var allowedGlobals = makeMap(
      'Infinity,undefined,NaN,isFinite,isNaN,' +
      'parseFloat,parseInt,decodeURI,decodeURIComponent,encodeURI,encodeURIComponent,' +
      'Math,Number,Date,Array,Object,Boolean,String,RegExp,Map,Set,JSON,Intl,' +
      'require' // for Webpack/Browserify
    );

    var warnNonPresent = function (target, key) {
      warn(
        "Property or method \"" + key + "\" is not defined on the instance but " +
        'referenced during render. Make sure that this property is reactive, ' +
        'either in the data option, or for class-based components, by ' +
        'initializing the property. ' +
        'See: https://vuejs.org/v2/guide/reactivity.html#Declaring-Reactive-Properties.',
        target
      );
    };

    var warnReservedPrefix = function (target, key) {
      warn(
        "Property \"" + key + "\" must be accessed with \"$data." + key + "\" because " +
        'properties starting with "$" or "_" are not proxied in the Vue instance to ' +
        'prevent conflicts with Vue internals' +
        'See: https://vuejs.org/v2/api/#data',
        target
      );
    };

    var hasProxy =
      typeof Proxy !== 'undefined' && isNative(Proxy);

    if (hasProxy) {
      var isBuiltInModifier = makeMap('stop,prevent,self,ctrl,shift,alt,meta,exact');
      config.keyCodes = new Proxy(config.keyCodes, {
        set: function set (target, key, value) {
          if (isBuiltInModifier(key)) {
            warn(("Avoid overwriting built-in modifier in config.keyCodes: ." + key));
            return false
          } else {
            target[key] = value;
            return true
          }
        }
      });
    }

    var hasHandler = {
      has: function has (target, key) {
        var has = key in target;
        var isAllowed = allowedGlobals(key) ||
          (typeof key === 'string' && key.charAt(0) === '_' && !(key in target.$data));
        if (!has && !isAllowed) {
          if (key in target.$data) { warnReservedPrefix(target, key); }
          else { warnNonPresent(target, key); }
        }
        return has || !isAllowed
      }
    };

    var getHandler = {
      get: function get (target, key) {
        if (typeof key === 'string' && !(key in target)) {
          if (key in target.$data) { warnReservedPrefix(target, key); }
          else { warnNonPresent(target, key); }
        }
        return target[key]
      }
    };

    initProxy = function initProxy (vm) {
      if (hasProxy) {
        // determine which proxy handler to use
        var options = vm.$options;
        var handlers = options.render && options.render._withStripped
          ? getHandler
          : hasHandler;
        vm._renderProxy = new Proxy(vm, handlers);
      } else {
        vm._renderProxy = vm;
      }
    };
  }

  /*  */

  var seenObjects = new _Set();

  /**
   * Recursively traverse an object to evoke all converted
   * getters, so that every nested property inside the object
   * is collected as a "deep" dependency.
   */
  function traverse (val) {
    _traverse(val, seenObjects);
    seenObjects.clear();
  }

  function _traverse (val, seen) {
    var i, keys;
    var isA = Array.isArray(val);
    if ((!isA && !isObject(val)) || Object.isFrozen(val) || val instanceof VNode) {
      return
    }
    if (val.__ob__) {
      var depId = val.__ob__.dep.id;
      if (seen.has(depId)) {
        return
      }
      seen.add(depId);
    }
    if (isA) {
      i = val.length;
      while (i--) { _traverse(val[i], seen); }
    } else {
      keys = Object.keys(val);
      i = keys.length;
      while (i--) { _traverse(val[keys[i]], seen); }
    }
  }

  var mark;
  var measure;

  {
    var perf = inBrowser && window.performance;
    /* istanbul ignore if */
    if (
      perf &&
      perf.mark &&
      perf.measure &&
      perf.clearMarks &&
      perf.clearMeasures
    ) {
      mark = function (tag) { return perf.mark(tag); };
      measure = function (name, startTag, endTag) {
        perf.measure(name, startTag, endTag);
        perf.clearMarks(startTag);
        perf.clearMarks(endTag);
        // perf.clearMeasures(name)
      };
    }
  }

  /*  */

  var normalizeEvent = cached(function (name) {
    var passive = name.charAt(0) === '&';
    name = passive ? name.slice(1) : name;
    var once$$1 = name.charAt(0) === '~'; // Prefixed last, checked first
    name = once$$1 ? name.slice(1) : name;
    var capture = name.charAt(0) === '!';
    name = capture ? name.slice(1) : name;
    return {
      name: name,
      once: once$$1,
      capture: capture,
      passive: passive
    }
  });

  function createFnInvoker (fns, vm) {
    function invoker () {
      var arguments$1 = arguments;

      var fns = invoker.fns;
      if (Array.isArray(fns)) {
        var cloned = fns.slice();
        for (var i = 0; i < cloned.length; i++) {
          invokeWithErrorHandling(cloned[i], null, arguments$1, vm, "v-on handler");
        }
      } else {
        // return handler return value for single handlers
        return invokeWithErrorHandling(fns, null, arguments, vm, "v-on handler")
      }
    }
    invoker.fns = fns;
    return invoker
  }

  function updateListeners (
    on,
    oldOn,
    add,
    remove$$1,
    createOnceHandler,
    vm
  ) {
    var name, def$$1, cur, old, event;
    for (name in on) {
      def$$1 = cur = on[name];
      old = oldOn[name];
      event = normalizeEvent(name);
      if (isUndef(cur)) {
        warn(
          "Invalid handler for event \"" + (event.name) + "\": got " + String(cur),
          vm
        );
      } else if (isUndef(old)) {
        if (isUndef(cur.fns)) {
          cur = on[name] = createFnInvoker(cur, vm);
        }
        if (isTrue(event.once)) {
          cur = on[name] = createOnceHandler(event.name, cur, event.capture);
        }
        add(event.name, cur, event.capture, event.passive, event.params);
      } else if (cur !== old) {
        old.fns = cur;
        on[name] = old;
      }
    }
    for (name in oldOn) {
      if (isUndef(on[name])) {
        event = normalizeEvent(name);
        remove$$1(event.name, oldOn[name], event.capture);
      }
    }
  }

  /*  */

  function mergeVNodeHook (def, hookKey, hook) {
    if (def instanceof VNode) {
      def = def.data.hook || (def.data.hook = {});
    }
    var invoker;
    var oldHook = def[hookKey];

    function wrappedHook () {
      hook.apply(this, arguments);
      // important: remove merged hook to ensure it's called only once
      // and prevent memory leak
      remove(invoker.fns, wrappedHook);
    }

    if (isUndef(oldHook)) {
      // no existing hook
      invoker = createFnInvoker([wrappedHook]);
    } else {
      /* istanbul ignore if */
      if (isDef(oldHook.fns) && isTrue(oldHook.merged)) {
        // already a merged invoker
        invoker = oldHook;
        invoker.fns.push(wrappedHook);
      } else {
        // existing plain hook
        invoker = createFnInvoker([oldHook, wrappedHook]);
      }
    }

    invoker.merged = true;
    def[hookKey] = invoker;
  }

  /*  */

  function extractPropsFromVNodeData (
    data,
    Ctor,
    tag
  ) {
    // we are only extracting raw values here.
    // validation and default values are handled in the child
    // component itself.
    var propOptions = Ctor.options.props;
    if (isUndef(propOptions)) {
      return
    }
    var res = {};
    var attrs = data.attrs;
    var props = data.props;
    if (isDef(attrs) || isDef(props)) {
      for (var key in propOptions) {
        var altKey = hyphenate(key);
        {
          var keyInLowerCase = key.toLowerCase();
          if (
            key !== keyInLowerCase &&
            attrs && hasOwn(attrs, keyInLowerCase)
          ) {
            tip(
              "Prop \"" + keyInLowerCase + "\" is passed to component " +
              (formatComponentName(tag || Ctor)) + ", but the declared prop name is" +
              " \"" + key + "\". " +
              "Note that HTML attributes are case-insensitive and camelCased " +
              "props need to use their kebab-case equivalents when using in-DOM " +
              "templates. You should probably use \"" + altKey + "\" instead of \"" + key + "\"."
            );
          }
        }
        checkProp(res, props, key, altKey, true) ||
        checkProp(res, attrs, key, altKey, false);
      }
    }
    return res
  }

  function checkProp (
    res,
    hash,
    key,
    altKey,
    preserve
  ) {
    if (isDef(hash)) {
      if (hasOwn(hash, key)) {
        res[key] = hash[key];
        if (!preserve) {
          delete hash[key];
        }
        return true
      } else if (hasOwn(hash, altKey)) {
        res[key] = hash[altKey];
        if (!preserve) {
          delete hash[altKey];
        }
        return true
      }
    }
    return false
  }

  /*  */

  // The template compiler attempts to minimize the need for normalization by
  // statically analyzing the template at compile time.
  //
  // For plain HTML markup, normalization can be completely skipped because the
  // generated render function is guaranteed to return Array<VNode>. There are
  // two cases where extra normalization is needed:

  // 1. When the children contains components - because a functional component
  // may return an Array instead of a single root. In this case, just a simple
  // normalization is needed - if any child is an Array, we flatten the whole
  // thing with Array.prototype.concat. It is guaranteed to be only 1-level deep
  // because functional components already normalize their own children.
  function simpleNormalizeChildren (children) {
    for (var i = 0; i < children.length; i++) {
      if (Array.isArray(children[i])) {
        return Array.prototype.concat.apply([], children)
      }
    }
    return children
  }

  // 2. When the children contains constructs that always generated nested Arrays,
  // e.g. <template>, <slot>, v-for, or when the children is provided by user
  // with hand-written render functions / JSX. In such cases a full normalization
  // is needed to cater to all possible types of children values.
  function normalizeChildren (children) {
    return isPrimitive(children)
      ? [createTextVNode(children)]
      : Array.isArray(children)
        ? normalizeArrayChildren(children)
        : undefined
  }

  function isTextNode (node) {
    return isDef(node) && isDef(node.text) && isFalse(node.isComment)
  }

  function normalizeArrayChildren (children, nestedIndex) {
    var res = [];
    var i, c, lastIndex, last;
    for (i = 0; i < children.length; i++) {
      c = children[i];
      if (isUndef(c) || typeof c === 'boolean') { continue }
      lastIndex = res.length - 1;
      last = res[lastIndex];
      //  nested
      if (Array.isArray(c)) {
        if (c.length > 0) {
          c = normalizeArrayChildren(c, ((nestedIndex || '') + "_" + i));
          // merge adjacent text nodes
          if (isTextNode(c[0]) && isTextNode(last)) {
            res[lastIndex] = createTextVNode(last.text + (c[0]).text);
            c.shift();
          }
          res.push.apply(res, c);
        }
      } else if (isPrimitive(c)) {
        if (isTextNode(last)) {
          // merge adjacent text nodes
          // this is necessary for SSR hydration because text nodes are
          // essentially merged when rendered to HTML strings
          res[lastIndex] = createTextVNode(last.text + c);
        } else if (c !== '') {
          // convert primitive to vnode
          res.push(createTextVNode(c));
        }
      } else {
        if (isTextNode(c) && isTextNode(last)) {
          // merge adjacent text nodes
          res[lastIndex] = createTextVNode(last.text + c.text);
        } else {
          // default key for nested array children (likely generated by v-for)
          if (isTrue(children._isVList) &&
            isDef(c.tag) &&
            isUndef(c.key) &&
            isDef(nestedIndex)) {
            c.key = "__vlist" + nestedIndex + "_" + i + "__";
          }
          res.push(c);
        }
      }
    }
    return res
  }

  /*  */

  function initProvide (vm) {
    var provide = vm.$options.provide;
    if (provide) {
      vm._provided = typeof provide === 'function'
        ? provide.call(vm)
        : provide;
    }
  }

  function initInjections (vm) {
    var result = resolveInject(vm.$options.inject, vm);
    if (result) {
      toggleObserving(false);
      Object.keys(result).forEach(function (key) {
        /* istanbul ignore else */
        {
          defineReactive$$1(vm, key, result[key], function () {
            warn(
              "Avoid mutating an injected value directly since the changes will be " +
              "overwritten whenever the provided component re-renders. " +
              "injection being mutated: \"" + key + "\"",
              vm
            );
          });
        }
      });
      toggleObserving(true);
    }
  }

  function resolveInject (inject, vm) {
    if (inject) {
      // inject is :any because flow is not smart enough to figure out cached
      var result = Object.create(null);
      var keys = hasSymbol
        ? Reflect.ownKeys(inject)
        : Object.keys(inject);

      for (var i = 0; i < keys.length; i++) {
        var key = keys[i];
        // #6574 in case the inject object is observed...
        if (key === '__ob__') { continue }
        var provideKey = inject[key].from;
        var source = vm;
        while (source) {
          if (source._provided && hasOwn(source._provided, provideKey)) {
            result[key] = source._provided[provideKey];
            break
          }
          source = source.$parent;
        }
        if (!source) {
          if ('default' in inject[key]) {
            var provideDefault = inject[key].default;
            result[key] = typeof provideDefault === 'function'
              ? provideDefault.call(vm)
              : provideDefault;
          } else {
            warn(("Injection \"" + key + "\" not found"), vm);
          }
        }
      }
      return result
    }
  }

  /*  */



  /**
   * Runtime helper for resolving raw children VNodes into a slot object.
   */
  function resolveSlots (
    children,
    context
  ) {
    if (!children || !children.length) {
      return {}
    }
    var slots = {};
    for (var i = 0, l = children.length; i < l; i++) {
      var child = children[i];
      var data = child.data;
      // remove slot attribute if the node is resolved as a Vue slot node
      if (data && data.attrs && data.attrs.slot) {
        delete data.attrs.slot;
      }
      // named slots should only be respected if the vnode was rendered in the
      // same context.
      if ((child.context === context || child.fnContext === context) &&
        data && data.slot != null
      ) {
        var name = data.slot;
        var slot = (slots[name] || (slots[name] = []));
        if (child.tag === 'template') {
          slot.push.apply(slot, child.children || []);
        } else {
          slot.push(child);
        }
      } else {
        (slots.default || (slots.default = [])).push(child);
      }
    }
    // ignore slots that contains only whitespace
    for (var name$1 in slots) {
      if (slots[name$1].every(isWhitespace)) {
        delete slots[name$1];
      }
    }
    return slots
  }

  function isWhitespace (node) {
    return (node.isComment && !node.asyncFactory) || node.text === ' '
  }

  /*  */

  function normalizeScopedSlots (
    slots,
    normalSlots,
    prevSlots
  ) {
    var res;
    var hasNormalSlots = Object.keys(normalSlots).length > 0;
    var isStable = slots ? !!slots.$stable : !hasNormalSlots;
    var key = slots && slots.$key;
    if (!slots) {
      res = {};
    } else if (slots._normalized) {
      // fast path 1: child component re-render only, parent did not change
      return slots._normalized
    } else if (
      isStable &&
      prevSlots &&
      prevSlots !== emptyObject &&
      key === prevSlots.$key &&
      !hasNormalSlots &&
      !prevSlots.$hasNormal
    ) {
      // fast path 2: stable scoped slots w/ no normal slots to proxy,
      // only need to normalize once
      return prevSlots
    } else {
      res = {};
      for (var key$1 in slots) {
        if (slots[key$1] && key$1[0] !== '$') {
          res[key$1] = normalizeScopedSlot(normalSlots, key$1, slots[key$1]);
        }
      }
    }
    // expose normal slots on scopedSlots
    for (var key$2 in normalSlots) {
      if (!(key$2 in res)) {
        res[key$2] = proxyNormalSlot(normalSlots, key$2);
      }
    }
    // avoriaz seems to mock a non-extensible $scopedSlots object
    // and when that is passed down this would cause an error
    if (slots && Object.isExtensible(slots)) {
      (slots)._normalized = res;
    }
    def(res, '$stable', isStable);
    def(res, '$key', key);
    def(res, '$hasNormal', hasNormalSlots);
    return res
  }

  function normalizeScopedSlot(normalSlots, key, fn) {
    var normalized = function () {
      var res = arguments.length ? fn.apply(null, arguments) : fn({});
      res = res && typeof res === 'object' && !Array.isArray(res)
        ? [res] // single vnode
        : normalizeChildren(res);
      return res && (
        res.length === 0 ||
        (res.length === 1 && res[0].isComment) // #9658
      ) ? undefined
        : res
    };
    // this is a slot using the new v-slot syntax without scope. although it is
    // compiled as a scoped slot, render fn users would expect it to be present
    // on this.$slots because the usage is semantically a normal slot.
    if (fn.proxy) {
      Object.defineProperty(normalSlots, key, {
        get: normalized,
        enumerable: true,
        configurable: true
      });
    }
    return normalized
  }

  function proxyNormalSlot(slots, key) {
    return function () { return slots[key]; }
  }

  /*  */

  /**
   * Runtime helper for rendering v-for lists.
   */
  function renderList (
    val,
    render
  ) {
    var ret, i, l, keys, key;
    if (Array.isArray(val) || typeof val === 'string') {
      ret = new Array(val.length);
      for (i = 0, l = val.length; i < l; i++) {
        ret[i] = render(val[i], i);
      }
    } else if (typeof val === 'number') {
      ret = new Array(val);
      for (i = 0; i < val; i++) {
        ret[i] = render(i + 1, i);
      }
    } else if (isObject(val)) {
      if (hasSymbol && val[Symbol.iterator]) {
        ret = [];
        var iterator = val[Symbol.iterator]();
        var result = iterator.next();
        while (!result.done) {
          ret.push(render(result.value, ret.length));
          result = iterator.next();
        }
      } else {
        keys = Object.keys(val);
        ret = new Array(keys.length);
        for (i = 0, l = keys.length; i < l; i++) {
          key = keys[i];
          ret[i] = render(val[key], key, i);
        }
      }
    }
    if (!isDef(ret)) {
      ret = [];
    }
    (ret)._isVList = true;
    return ret
  }

  /*  */

  /**
   * Runtime helper for rendering <slot>
   */
  function renderSlot (
    name,
    fallback,
    props,
    bindObject
  ) {
    var scopedSlotFn = this.$scopedSlots[name];
    var nodes;
    if (scopedSlotFn) { // scoped slot
      props = props || {};
      if (bindObject) {
        if (!isObject(bindObject)) {
          warn(
            'slot v-bind without argument expects an Object',
            this
          );
        }
        props = extend(extend({}, bindObject), props);
      }
      nodes = scopedSlotFn(props) || fallback;
    } else {
      nodes = this.$slots[name] || fallback;
    }

    var target = props && props.slot;
    if (target) {
      return this.$createElement('template', { slot: target }, nodes)
    } else {
      return nodes
    }
  }

  /*  */

  /**
   * Runtime helper for resolving filters
   */
  function resolveFilter (id) {
    return resolveAsset(this.$options, 'filters', id, true) || identity
  }

  /*  */

  function isKeyNotMatch (expect, actual) {
    if (Array.isArray(expect)) {
      return expect.indexOf(actual) === -1
    } else {
      return expect !== actual
    }
  }

  /**
   * Runtime helper for checking keyCodes from config.
   * exposed as Vue.prototype._k
   * passing in eventKeyName as last argument separately for backwards compat
   */
  function checkKeyCodes (
    eventKeyCode,
    key,
    builtInKeyCode,
    eventKeyName,
    builtInKeyName
  ) {
    var mappedKeyCode = config.keyCodes[key] || builtInKeyCode;
    if (builtInKeyName && eventKeyName && !config.keyCodes[key]) {
      return isKeyNotMatch(builtInKeyName, eventKeyName)
    } else if (mappedKeyCode) {
      return isKeyNotMatch(mappedKeyCode, eventKeyCode)
    } else if (eventKeyName) {
      return hyphenate(eventKeyName) !== key
    }
  }

  /*  */

  /**
   * Runtime helper for merging v-bind="object" into a VNode's data.
   */
  function bindObjectProps (
    data,
    tag,
    value,
    asProp,
    isSync
  ) {
    if (value) {
      if (!isObject(value)) {
        warn(
          'v-bind without argument expects an Object or Array value',
          this
        );
      } else {
        if (Array.isArray(value)) {
          value = toObject(value);
        }
        var hash;
        var loop = function ( key ) {
          if (
            key === 'class' ||
            key === 'style' ||
            isReservedAttribute(key)
          ) {
            hash = data;
          } else {
            var type = data.attrs && data.attrs.type;
            hash = asProp || config.mustUseProp(tag, type, key)
              ? data.domProps || (data.domProps = {})
              : data.attrs || (data.attrs = {});
          }
          var camelizedKey = camelize(key);
          var hyphenatedKey = hyphenate(key);
          if (!(camelizedKey in hash) && !(hyphenatedKey in hash)) {
            hash[key] = value[key];

            if (isSync) {
              var on = data.on || (data.on = {});
              on[("update:" + key)] = function ($event) {
                value[key] = $event;
              };
            }
          }
        };

        for (var key in value) loop( key );
      }
    }
    return data
  }

  /*  */

  /**
   * Runtime helper for rendering static trees.
   */
  function renderStatic (
    index,
    isInFor
  ) {
    var cached = this._staticTrees || (this._staticTrees = []);
    var tree = cached[index];
    // if has already-rendered static tree and not inside v-for,
    // we can reuse the same tree.
    if (tree && !isInFor) {
      return tree
    }
    // otherwise, render a fresh tree.
    tree = cached[index] = this.$options.staticRenderFns[index].call(
      this._renderProxy,
      null,
      this // for render fns generated for functional component templates
    );
    markStatic(tree, ("__static__" + index), false);
    return tree
  }

  /**
   * Runtime helper for v-once.
   * Effectively it means marking the node as static with a unique key.
   */
  function markOnce (
    tree,
    index,
    key
  ) {
    markStatic(tree, ("__once__" + index + (key ? ("_" + key) : "")), true);
    return tree
  }

  function markStatic (
    tree,
    key,
    isOnce
  ) {
    if (Array.isArray(tree)) {
      for (var i = 0; i < tree.length; i++) {
        if (tree[i] && typeof tree[i] !== 'string') {
          markStaticNode(tree[i], (key + "_" + i), isOnce);
        }
      }
    } else {
      markStaticNode(tree, key, isOnce);
    }
  }

  function markStaticNode (node, key, isOnce) {
    node.isStatic = true;
    node.key = key;
    node.isOnce = isOnce;
  }

  /*  */

  function bindObjectListeners (data, value) {
    if (value) {
      if (!isPlainObject(value)) {
        warn(
          'v-on without argument expects an Object value',
          this
        );
      } else {
        var on = data.on = data.on ? extend({}, data.on) : {};
        for (var key in value) {
          var existing = on[key];
          var ours = value[key];
          on[key] = existing ? [].concat(existing, ours) : ours;
        }
      }
    }
    return data
  }

  /*  */

  function resolveScopedSlots (
    fns, // see flow/vnode
    res,
    // the following are added in 2.6
    hasDynamicKeys,
    contentHashKey
  ) {
    res = res || { $stable: !hasDynamicKeys };
    for (var i = 0; i < fns.length; i++) {
      var slot = fns[i];
      if (Array.isArray(slot)) {
        resolveScopedSlots(slot, res, hasDynamicKeys);
      } else if (slot) {
        // marker for reverse proxying v-slot without scope on this.$slots
        if (slot.proxy) {
          slot.fn.proxy = true;
        }
        res[slot.key] = slot.fn;
      }
    }
    if (contentHashKey) {
      (res).$key = contentHashKey;
    }
    return res
  }

  /*  */

  function bindDynamicKeys (baseObj, values) {
    for (var i = 0; i < values.length; i += 2) {
      var key = values[i];
      if (typeof key === 'string' && key) {
        baseObj[values[i]] = values[i + 1];
      } else if (key !== '' && key !== null) {
        // null is a speical value for explicitly removing a binding
        warn(
          ("Invalid value for dynamic directive argument (expected string or null): " + key),
          this
        );
      }
    }
    return baseObj
  }

  // helper to dynamically append modifier runtime markers to event names.
  // ensure only append when value is already string, otherwise it will be cast
  // to string and cause the type check to miss.
  function prependModifier (value, symbol) {
    return typeof value === 'string' ? symbol + value : value
  }

  /*  */

  function installRenderHelpers (target) {
    target._o = markOnce;
    target._n = toNumber;
    target._s = toString;
    target._l = renderList;
    target._t = renderSlot;
    target._q = looseEqual;
    target._i = looseIndexOf;
    target._m = renderStatic;
    target._f = resolveFilter;
    target._k = checkKeyCodes;
    target._b = bindObjectProps;
    target._v = createTextVNode;
    target._e = createEmptyVNode;
    target._u = resolveScopedSlots;
    target._g = bindObjectListeners;
    target._d = bindDynamicKeys;
    target._p = prependModifier;
  }

  /*  */

  function FunctionalRenderContext (
    data,
    props,
    children,
    parent,
    Ctor
  ) {
    var this$1 = this;

    var options = Ctor.options;
    // ensure the createElement function in functional components
    // gets a unique context - this is necessary for correct named slot check
    var contextVm;
    if (hasOwn(parent, '_uid')) {
      contextVm = Object.create(parent);
      // $flow-disable-line
      contextVm._original = parent;
    } else {
      // the context vm passed in is a functional context as well.
      // in this case we want to make sure we are able to get a hold to the
      // real context instance.
      contextVm = parent;
      // $flow-disable-line
      parent = parent._original;
    }
    var isCompiled = isTrue(options._compiled);
    var needNormalization = !isCompiled;

    this.data = data;
    this.props = props;
    this.children = children;
    this.parent = parent;
    this.listeners = data.on || emptyObject;
    this.injections = resolveInject(options.inject, parent);
    this.slots = function () {
      if (!this$1.$slots) {
        normalizeScopedSlots(
          data.scopedSlots,
          this$1.$slots = resolveSlots(children, parent)
        );
      }
      return this$1.$slots
    };

    Object.defineProperty(this, 'scopedSlots', ({
      enumerable: true,
      get: function get () {
        return normalizeScopedSlots(data.scopedSlots, this.slots())
      }
    }));

    // support for compiled functional template
    if (isCompiled) {
      // exposing $options for renderStatic()
      this.$options = options;
      // pre-resolve slots for renderSlot()
      this.$slots = this.slots();
      this.$scopedSlots = normalizeScopedSlots(data.scopedSlots, this.$slots);
    }

    if (options._scopeId) {
      this._c = function (a, b, c, d) {
        var vnode = createElement(contextVm, a, b, c, d, needNormalization);
        if (vnode && !Array.isArray(vnode)) {
          vnode.fnScopeId = options._scopeId;
          vnode.fnContext = parent;
        }
        return vnode
      };
    } else {
      this._c = function (a, b, c, d) { return createElement(contextVm, a, b, c, d, needNormalization); };
    }
  }

  installRenderHelpers(FunctionalRenderContext.prototype);

  function createFunctionalComponent (
    Ctor,
    propsData,
    data,
    contextVm,
    children
  ) {
    var options = Ctor.options;
    var props = {};
    var propOptions = options.props;
    if (isDef(propOptions)) {
      for (var key in propOptions) {
        props[key] = validateProp(key, propOptions, propsData || emptyObject);
      }
    } else {
      if (isDef(data.attrs)) { mergeProps(props, data.attrs); }
      if (isDef(data.props)) { mergeProps(props, data.props); }
    }

    var renderContext = new FunctionalRenderContext(
      data,
      props,
      children,
      contextVm,
      Ctor
    );

    var vnode = options.render.call(null, renderContext._c, renderContext);

    if (vnode instanceof VNode) {
      return cloneAndMarkFunctionalResult(vnode, data, renderContext.parent, options, renderContext)
    } else if (Array.isArray(vnode)) {
      var vnodes = normalizeChildren(vnode) || [];
      var res = new Array(vnodes.length);
      for (var i = 0; i < vnodes.length; i++) {
        res[i] = cloneAndMarkFunctionalResult(vnodes[i], data, renderContext.parent, options, renderContext);
      }
      return res
    }
  }

  function cloneAndMarkFunctionalResult (vnode, data, contextVm, options, renderContext) {
    // #7817 clone node before setting fnContext, otherwise if the node is reused
    // (e.g. it was from a cached normal slot) the fnContext causes named slots
    // that should not be matched to match.
    var clone = cloneVNode(vnode);
    clone.fnContext = contextVm;
    clone.fnOptions = options;
    {
      (clone.devtoolsMeta = clone.devtoolsMeta || {}).renderContext = renderContext;
    }
    if (data.slot) {
      (clone.data || (clone.data = {})).slot = data.slot;
    }
    return clone
  }

  function mergeProps (to, from) {
    for (var key in from) {
      to[camelize(key)] = from[key];
    }
  }

  /*  */

  /*  */

  /*  */

  /*  */

  // inline hooks to be invoked on component VNodes during patch
  var componentVNodeHooks = {
    init: function init (vnode, hydrating) {
      if (
        vnode.componentInstance &&
        !vnode.componentInstance._isDestroyed &&
        vnode.data.keepAlive
      ) {
        // kept-alive components, treat as a patch
        var mountedNode = vnode; // work around flow
        componentVNodeHooks.prepatch(mountedNode, mountedNode);
      } else {
        var child = vnode.componentInstance = createComponentInstanceForVnode(
          vnode,
          activeInstance
        );
        child.$mount(hydrating ? vnode.elm : undefined, hydrating);
      }
    },

    prepatch: function prepatch (oldVnode, vnode) {
      var options = vnode.componentOptions;
      var child = vnode.componentInstance = oldVnode.componentInstance;
      updateChildComponent(
        child,
        options.propsData, // updated props
        options.listeners, // updated listeners
        vnode, // new parent vnode
        options.children // new children
      );
    },

    insert: function insert (vnode) {
      var context = vnode.context;
      var componentInstance = vnode.componentInstance;
      if (!componentInstance._isMounted) {
        componentInstance._isMounted = true;
        callHook(componentInstance, 'mounted');
      }
      if (vnode.data.keepAlive) {
        if (context._isMounted) {
          // vue-router#1212
          // During updates, a kept-alive component's child components may
          // change, so directly walking the tree here may call activated hooks
          // on incorrect children. Instead we push them into a queue which will
          // be processed after the whole patch process ended.
          queueActivatedComponent(componentInstance);
        } else {
          activateChildComponent(componentInstance, true /* direct */);
        }
      }
    },

    destroy: function destroy (vnode) {
      var componentInstance = vnode.componentInstance;
      if (!componentInstance._isDestroyed) {
        if (!vnode.data.keepAlive) {
          componentInstance.$destroy();
        } else {
          deactivateChildComponent(componentInstance, true /* direct */);
        }
      }
    }
  };

  var hooksToMerge = Object.keys(componentVNodeHooks);

  function createComponent (
    Ctor,
    data,
    context,
    children,
    tag
  ) {
    if (isUndef(Ctor)) {
      return
    }

    var baseCtor = context.$options._base;

    // plain options object: turn it into a constructor
    if (isObject(Ctor)) {
      Ctor = baseCtor.extend(Ctor);
    }

    // if at this stage it's not a constructor or an async component factory,
    // reject.
    if (typeof Ctor !== 'function') {
      {
        warn(("Invalid Component definition: " + (String(Ctor))), context);
      }
      return
    }

    // async component
    var asyncFactory;
    if (isUndef(Ctor.cid)) {
      asyncFactory = Ctor;
      Ctor = resolveAsyncComponent(asyncFactory, baseCtor);
      if (Ctor === undefined) {
        // return a placeholder node for async component, which is rendered
        // as a comment node but preserves all the raw information for the node.
        // the information will be used for async server-rendering and hydration.
        return createAsyncPlaceholder(
          asyncFactory,
          data,
          context,
          children,
          tag
        )
      }
    }

    data = data || {};

    // resolve constructor options in case global mixins are applied after
    // component constructor creation
    resolveConstructorOptions(Ctor);

    // transform component v-model data into props & events
    if (isDef(data.model)) {
      transformModel(Ctor.options, data);
    }

    // extract props
    var propsData = extractPropsFromVNodeData(data, Ctor, tag);

    // functional component
    if (isTrue(Ctor.options.functional)) {
      return createFunctionalComponent(Ctor, propsData, data, context, children)
    }

    // extract listeners, since these needs to be treated as
    // child component listeners instead of DOM listeners
    var listeners = data.on;
    // replace with listeners with .native modifier
    // so it gets processed during parent component patch.
    data.on = data.nativeOn;

    if (isTrue(Ctor.options.abstract)) {
      // abstract components do not keep anything
      // other than props & listeners & slot

      // work around flow
      var slot = data.slot;
      data = {};
      if (slot) {
        data.slot = slot;
      }
    }

    // install component management hooks onto the placeholder node
    installComponentHooks(data);

    // return a placeholder vnode
    var name = Ctor.options.name || tag;
    var vnode = new VNode(
      ("vue-component-" + (Ctor.cid) + (name ? ("-" + name) : '')),
      data, undefined, undefined, undefined, context,
      { Ctor: Ctor, propsData: propsData, listeners: listeners, tag: tag, children: children },
      asyncFactory
    );

    return vnode
  }

  function createComponentInstanceForVnode (
    vnode, // we know it's MountedComponentVNode but flow doesn't
    parent // activeInstance in lifecycle state
  ) {
    var options = {
      _isComponent: true,
      _parentVnode: vnode,
      parent: parent
    };
    // check inline-template render functions
    var inlineTemplate = vnode.data.inlineTemplate;
    if (isDef(inlineTemplate)) {
      options.render = inlineTemplate.render;
      options.staticRenderFns = inlineTemplate.staticRenderFns;
    }
    return new vnode.componentOptions.Ctor(options)
  }

  function installComponentHooks (data) {
    var hooks = data.hook || (data.hook = {});
    for (var i = 0; i < hooksToMerge.length; i++) {
      var key = hooksToMerge[i];
      var existing = hooks[key];
      var toMerge = componentVNodeHooks[key];
      if (existing !== toMerge && !(existing && existing._merged)) {
        hooks[key] = existing ? mergeHook$1(toMerge, existing) : toMerge;
      }
    }
  }

  function mergeHook$1 (f1, f2) {
    var merged = function (a, b) {
      // flow complains about extra args which is why we use any
      f1(a, b);
      f2(a, b);
    };
    merged._merged = true;
    return merged
  }

  // transform component v-model info (value and callback) into
  // prop and event handler respectively.
  function transformModel (options, data) {
    var prop = (options.model && options.model.prop) || 'value';
    var event = (options.model && options.model.event) || 'input'
    ;(data.attrs || (data.attrs = {}))[prop] = data.model.value;
    var on = data.on || (data.on = {});
    var existing = on[event];
    var callback = data.model.callback;
    if (isDef(existing)) {
      if (
        Array.isArray(existing)
          ? existing.indexOf(callback) === -1
          : existing !== callback
      ) {
        on[event] = [callback].concat(existing);
      }
    } else {
      on[event] = callback;
    }
  }

  /*  */

  var SIMPLE_NORMALIZE = 1;
  var ALWAYS_NORMALIZE = 2;

  // wrapper function for providing a more flexible interface
  // without getting yelled at by flow
  function createElement (
    context,
    tag,
    data,
    children,
    normalizationType,
    alwaysNormalize
  ) {
    if (Array.isArray(data) || isPrimitive(data)) {
      normalizationType = children;
      children = data;
      data = undefined;
    }
    if (isTrue(alwaysNormalize)) {
      normalizationType = ALWAYS_NORMALIZE;
    }
    return _createElement(context, tag, data, children, normalizationType)
  }

  function _createElement (
    context,
    tag,
    data,
    children,
    normalizationType
  ) {
    if (isDef(data) && isDef((data).__ob__)) {
      warn(
        "Avoid using observed data object as vnode data: " + (JSON.stringify(data)) + "\n" +
        'Always create fresh vnode data objects in each render!',
        context
      );
      return createEmptyVNode()
    }
    // object syntax in v-bind
    if (isDef(data) && isDef(data.is)) {
      tag = data.is;
    }
    if (!tag) {
      // in case of component :is set to falsy value
      return createEmptyVNode()
    }
    // warn against non-primitive key
    if (isDef(data) && isDef(data.key) && !isPrimitive(data.key)
    ) {
      {
        warn(
          'Avoid using non-primitive value as key, ' +
          'use string/number value instead.',
          context
        );
      }
    }
    // support single function children as default scoped slot
    if (Array.isArray(children) &&
      typeof children[0] === 'function'
    ) {
      data = data || {};
      data.scopedSlots = { default: children[0] };
      children.length = 0;
    }
    if (normalizationType === ALWAYS_NORMALIZE) {
      children = normalizeChildren(children);
    } else if (normalizationType === SIMPLE_NORMALIZE) {
      children = simpleNormalizeChildren(children);
    }
    var vnode, ns;
    if (typeof tag === 'string') {
      var Ctor;
      ns = (context.$vnode && context.$vnode.ns) || config.getTagNamespace(tag);
      if (config.isReservedTag(tag)) {
        // platform built-in elements
        vnode = new VNode(
          config.parsePlatformTagName(tag), data, children,
          undefined, undefined, context
        );
      } else if ((!data || !data.pre) && isDef(Ctor = resolveAsset(context.$options, 'components', tag))) {
        // component
        vnode = createComponent(Ctor, data, context, children, tag);
      } else {
        // unknown or unlisted namespaced elements
        // check at runtime because it may get assigned a namespace when its
        // parent normalizes children
        vnode = new VNode(
          tag, data, children,
          undefined, undefined, context
        );
      }
    } else {
      // direct component options / constructor
      vnode = createComponent(tag, data, context, children);
    }
    if (Array.isArray(vnode)) {
      return vnode
    } else if (isDef(vnode)) {
      if (isDef(ns)) { applyNS(vnode, ns); }
      if (isDef(data)) { registerDeepBindings(data); }
      return vnode
    } else {
      return createEmptyVNode()
    }
  }

  function applyNS (vnode, ns, force) {
    vnode.ns = ns;
    if (vnode.tag === 'foreignObject') {
      // use default namespace inside foreignObject
      ns = undefined;
      force = true;
    }
    if (isDef(vnode.children)) {
      for (var i = 0, l = vnode.children.length; i < l; i++) {
        var child = vnode.children[i];
        if (isDef(child.tag) && (
          isUndef(child.ns) || (isTrue(force) && child.tag !== 'svg'))) {
          applyNS(child, ns, force);
        }
      }
    }
  }

  // ref #5318
  // necessary to ensure parent re-render when deep bindings like :style and
  // :class are used on slot nodes
  function registerDeepBindings (data) {
    if (isObject(data.style)) {
      traverse(data.style);
    }
    if (isObject(data.class)) {
      traverse(data.class);
    }
  }

  /*  */

  function initRender (vm) {
    vm._vnode = null; // the root of the child tree
    vm._staticTrees = null; // v-once cached trees
    var options = vm.$options;
    var parentVnode = vm.$vnode = options._parentVnode; // the placeholder node in parent tree
    var renderContext = parentVnode && parentVnode.context;
    vm.$slots = resolveSlots(options._renderChildren, renderContext);
    vm.$scopedSlots = emptyObject;
    // bind the createElement fn to this instance
    // so that we get proper render context inside it.
    // args order: tag, data, children, normalizationType, alwaysNormalize
    // internal version is used by render functions compiled from templates
    vm._c = function (a, b, c, d) { return createElement(vm, a, b, c, d, false); };
    // normalization is always applied for the public version, used in
    // user-written render functions.
    vm.$createElement = function (a, b, c, d) { return createElement(vm, a, b, c, d, true); };

    // $attrs & $listeners are exposed for easier HOC creation.
    // they need to be reactive so that HOCs using them are always updated
    var parentData = parentVnode && parentVnode.data;

    /* istanbul ignore else */
    {
      defineReactive$$1(vm, '$attrs', parentData && parentData.attrs || emptyObject, function () {
        !isUpdatingChildComponent && warn("$attrs is readonly.", vm);
      }, true);
      defineReactive$$1(vm, '$listeners', options._parentListeners || emptyObject, function () {
        !isUpdatingChildComponent && warn("$listeners is readonly.", vm);
      }, true);
    }
  }

  var currentRenderingInstance = null;

  function renderMixin (Vue) {
    // install runtime convenience helpers
    installRenderHelpers(Vue.prototype);

    Vue.prototype.$nextTick = function (fn) {
      return nextTick(fn, this)
    };

    Vue.prototype._render = function () {
      var vm = this;
      var ref = vm.$options;
      var render = ref.render;
      var _parentVnode = ref._parentVnode;

      if (_parentVnode) {
        vm.$scopedSlots = normalizeScopedSlots(
          _parentVnode.data.scopedSlots,
          vm.$slots,
          vm.$scopedSlots
        );
      }

      // set parent vnode. this allows render functions to have access
      // to the data on the placeholder node.
      vm.$vnode = _parentVnode;
      // render self
      var vnode;
      try {
        // There's no need to maintain a stack becaues all render fns are called
        // separately from one another. Nested component's render fns are called
        // when parent component is patched.
        currentRenderingInstance = vm;
        vnode = render.call(vm._renderProxy, vm.$createElement);
      } catch (e) {
        handleError(e, vm, "render");
        // return error render result,
        // or previous vnode to prevent render error causing blank component
        /* istanbul ignore else */
        if (vm.$options.renderError) {
          try {
            vnode = vm.$options.renderError.call(vm._renderProxy, vm.$createElement, e);
          } catch (e) {
            handleError(e, vm, "renderError");
            vnode = vm._vnode;
          }
        } else {
          vnode = vm._vnode;
        }
      } finally {
        currentRenderingInstance = null;
      }
      // if the returned array contains only a single node, allow it
      if (Array.isArray(vnode) && vnode.length === 1) {
        vnode = vnode[0];
      }
      // return empty vnode in case the render function errored out
      if (!(vnode instanceof VNode)) {
        if (Array.isArray(vnode)) {
          warn(
            'Multiple root nodes returned from render function. Render function ' +
            'should return a single root node.',
            vm
          );
        }
        vnode = createEmptyVNode();
      }
      // set parent
      vnode.parent = _parentVnode;
      return vnode
    };
  }

  /*  */

  function ensureCtor (comp, base) {
    if (
      comp.__esModule ||
      (hasSymbol && comp[Symbol.toStringTag] === 'Module')
    ) {
      comp = comp.default;
    }
    return isObject(comp)
      ? base.extend(comp)
      : comp
  }

  function createAsyncPlaceholder (
    factory,
    data,
    context,
    children,
    tag
  ) {
    var node = createEmptyVNode();
    node.asyncFactory = factory;
    node.asyncMeta = { data: data, context: context, children: children, tag: tag };
    return node
  }

  function resolveAsyncComponent (
    factory,
    baseCtor
  ) {
    if (isTrue(factory.error) && isDef(factory.errorComp)) {
      return factory.errorComp
    }

    if (isDef(factory.resolved)) {
      return factory.resolved
    }

    var owner = currentRenderingInstance;
    if (owner && isDef(factory.owners) && factory.owners.indexOf(owner) === -1) {
      // already pending
      factory.owners.push(owner);
    }

    if (isTrue(factory.loading) && isDef(factory.loadingComp)) {
      return factory.loadingComp
    }

    if (owner && !isDef(factory.owners)) {
      var owners = factory.owners = [owner];
      var sync = true;
      var timerLoading = null;
      var timerTimeout = null

      ;(owner).$on('hook:destroyed', function () { return remove(owners, owner); });

      var forceRender = function (renderCompleted) {
        for (var i = 0, l = owners.length; i < l; i++) {
          (owners[i]).$forceUpdate();
        }

        if (renderCompleted) {
          owners.length = 0;
          if (timerLoading !== null) {
            clearTimeout(timerLoading);
            timerLoading = null;
          }
          if (timerTimeout !== null) {
            clearTimeout(timerTimeout);
            timerTimeout = null;
          }
        }
      };

      var resolve = once(function (res) {
        // cache resolved
        factory.resolved = ensureCtor(res, baseCtor);
        // invoke callbacks only if this is not a synchronous resolve
        // (async resolves are shimmed as synchronous during SSR)
        if (!sync) {
          forceRender(true);
        } else {
          owners.length = 0;
        }
      });

      var reject = once(function (reason) {
        warn(
          "Failed to resolve async component: " + (String(factory)) +
          (reason ? ("\nReason: " + reason) : '')
        );
        if (isDef(factory.errorComp)) {
          factory.error = true;
          forceRender(true);
        }
      });

      var res = factory(resolve, reject);

      if (isObject(res)) {
        if (isPromise(res)) {
          // () => Promise
          if (isUndef(factory.resolved)) {
            res.then(resolve, reject);
          }
        } else if (isPromise(res.component)) {
          res.component.then(resolve, reject);

          if (isDef(res.error)) {
            factory.errorComp = ensureCtor(res.error, baseCtor);
          }

          if (isDef(res.loading)) {
            factory.loadingComp = ensureCtor(res.loading, baseCtor);
            if (res.delay === 0) {
              factory.loading = true;
            } else {
              timerLoading = setTimeout(function () {
                timerLoading = null;
                if (isUndef(factory.resolved) && isUndef(factory.error)) {
                  factory.loading = true;
                  forceRender(false);
                }
              }, res.delay || 200);
            }
          }

          if (isDef(res.timeout)) {
            timerTimeout = setTimeout(function () {
              timerTimeout = null;
              if (isUndef(factory.resolved)) {
                reject(
                  "timeout (" + (res.timeout) + "ms)"
                );
              }
            }, res.timeout);
          }
        }
      }

      sync = false;
      // return in case resolved synchronously
      return factory.loading
        ? factory.loadingComp
        : factory.resolved
    }
  }

  /*  */

  function isAsyncPlaceholder (node) {
    return node.isComment && node.asyncFactory
  }

  /*  */

  function getFirstComponentChild (children) {
    if (Array.isArray(children)) {
      for (var i = 0; i < children.length; i++) {
        var c = children[i];
        if (isDef(c) && (isDef(c.componentOptions) || isAsyncPlaceholder(c))) {
          return c
        }
      }
    }
  }

  /*  */

  /*  */

  function initEvents (vm) {
    vm._events = Object.create(null);
    vm._hasHookEvent = false;
    // init parent attached events
    var listeners = vm.$options._parentListeners;
    if (listeners) {
      updateComponentListeners(vm, listeners);
    }
  }

  var target;

  function add (event, fn) {
    target.$on(event, fn);
  }

  function remove$1 (event, fn) {
    target.$off(event, fn);
  }

  function createOnceHandler (event, fn) {
    var _target = target;
    return function onceHandler () {
      var res = fn.apply(null, arguments);
      if (res !== null) {
        _target.$off(event, onceHandler);
      }
    }
  }

  function updateComponentListeners (
    vm,
    listeners,
    oldListeners
  ) {
    target = vm;
    updateListeners(listeners, oldListeners || {}, add, remove$1, createOnceHandler, vm);
    target = undefined;
  }

  function eventsMixin (Vue) {
    var hookRE = /^hook:/;
    Vue.prototype.$on = function (event, fn) {
      var vm = this;
      if (Array.isArray(event)) {
        for (var i = 0, l = event.length; i < l; i++) {
          vm.$on(event[i], fn);
        }
      } else {
        (vm._events[event] || (vm._events[event] = [])).push(fn);
        // optimize hook:event cost by using a boolean flag marked at registration
        // instead of a hash lookup
        if (hookRE.test(event)) {
          vm._hasHookEvent = true;
        }
      }
      return vm
    };

    Vue.prototype.$once = function (event, fn) {
      var vm = this;
      function on () {
        vm.$off(event, on);
        fn.apply(vm, arguments);
      }
      on.fn = fn;
      vm.$on(event, on);
      return vm
    };

    Vue.prototype.$off = function (event, fn) {
      var vm = this;
      // all
      if (!arguments.length) {
        vm._events = Object.create(null);
        return vm
      }
      // array of events
      if (Array.isArray(event)) {
        for (var i$1 = 0, l = event.length; i$1 < l; i$1++) {
          vm.$off(event[i$1], fn);
        }
        return vm
      }
      // specific event
      var cbs = vm._events[event];
      if (!cbs) {
        return vm
      }
      if (!fn) {
        vm._events[event] = null;
        return vm
      }
      // specific handler
      var cb;
      var i = cbs.length;
      while (i--) {
        cb = cbs[i];
        if (cb === fn || cb.fn === fn) {
          cbs.splice(i, 1);
          break
        }
      }
      return vm
    };

    Vue.prototype.$emit = function (event) {
      var vm = this;
      {
        var lowerCaseEvent = event.toLowerCase();
        if (lowerCaseEvent !== event && vm._events[lowerCaseEvent]) {
          tip(
            "Event \"" + lowerCaseEvent + "\" is emitted in component " +
            (formatComponentName(vm)) + " but the handler is registered for \"" + event + "\". " +
            "Note that HTML attributes are case-insensitive and you cannot use " +
            "v-on to listen to camelCase events when using in-DOM templates. " +
            "You should probably use \"" + (hyphenate(event)) + "\" instead of \"" + event + "\"."
          );
        }
      }
      var cbs = vm._events[event];
      if (cbs) {
        cbs = cbs.length > 1 ? toArray(cbs) : cbs;
        var args = toArray(arguments, 1);
        var info = "event handler for \"" + event + "\"";
        for (var i = 0, l = cbs.length; i < l; i++) {
          invokeWithErrorHandling(cbs[i], vm, args, vm, info);
        }
      }
      return vm
    };
  }

  /*  */

  var activeInstance = null;
  var isUpdatingChildComponent = false;

  function setActiveInstance(vm) {
    var prevActiveInstance = activeInstance;
    activeInstance = vm;
    return function () {
      activeInstance = prevActiveInstance;
    }
  }

  function initLifecycle (vm) {
    var options = vm.$options;

    // locate first non-abstract parent
    var parent = options.parent;
    if (parent && !options.abstract) {
      while (parent.$options.abstract && parent.$parent) {
        parent = parent.$parent;
      }
      parent.$children.push(vm);
    }

    vm.$parent = parent;
    vm.$root = parent ? parent.$root : vm;

    vm.$children = [];
    vm.$refs = {};

    vm._watcher = null;
    vm._inactive = null;
    vm._directInactive = false;
    vm._isMounted = false;
    vm._isDestroyed = false;
    vm._isBeingDestroyed = false;
  }

  function lifecycleMixin (Vue) {
    Vue.prototype._update = function (vnode, hydrating) {
      var vm = this;
      var prevEl = vm.$el;
      var prevVnode = vm._vnode;
      var restoreActiveInstance = setActiveInstance(vm);
      vm._vnode = vnode;
      // Vue.prototype.__patch__ is injected in entry points
      // based on the rendering backend used.
      if (!prevVnode) {
        // initial render
        vm.$el = vm.__patch__(vm.$el, vnode, hydrating, false /* removeOnly */);
      } else {
        // updates
        vm.$el = vm.__patch__(prevVnode, vnode);
      }
      restoreActiveInstance();
      // update __vue__ reference
      if (prevEl) {
        prevEl.__vue__ = null;
      }
      if (vm.$el) {
        vm.$el.__vue__ = vm;
      }
      // if parent is an HOC, update its $el as well
      if (vm.$vnode && vm.$parent && vm.$vnode === vm.$parent._vnode) {
        vm.$parent.$el = vm.$el;
      }
      // updated hook is called by the scheduler to ensure that children are
      // updated in a parent's updated hook.
    };

    Vue.prototype.$forceUpdate = function () {
      var vm = this;
      if (vm._watcher) {
        vm._watcher.update();
      }
    };

    Vue.prototype.$destroy = function () {
      var vm = this;
      if (vm._isBeingDestroyed) {
        return
      }
      callHook(vm, 'beforeDestroy');
      vm._isBeingDestroyed = true;
      // remove self from parent
      var parent = vm.$parent;
      if (parent && !parent._isBeingDestroyed && !vm.$options.abstract) {
        remove(parent.$children, vm);
      }
      // teardown watchers
      if (vm._watcher) {
        vm._watcher.teardown();
      }
      var i = vm._watchers.length;
      while (i--) {
        vm._watchers[i].teardown();
      }
      // remove reference from data ob
      // frozen object may not have observer.
      if (vm._data.__ob__) {
        vm._data.__ob__.vmCount--;
      }
      // call the last hook...
      vm._isDestroyed = true;
      // invoke destroy hooks on current rendered tree
      vm.__patch__(vm._vnode, null);
      // fire destroyed hook
      callHook(vm, 'destroyed');
      // turn off all instance listeners.
      vm.$off();
      // remove __vue__ reference
      if (vm.$el) {
        vm.$el.__vue__ = null;
      }
      // release circular reference (#6759)
      if (vm.$vnode) {
        vm.$vnode.parent = null;
      }
    };
  }

  function mountComponent (
    vm,
    el,
    hydrating
  ) {
    vm.$el = el;
    if (!vm.$options.render) {
      vm.$options.render = createEmptyVNode;
      {
        /* istanbul ignore if */
        if ((vm.$options.template && vm.$options.template.charAt(0) !== '#') ||
          vm.$options.el || el) {
          warn(
            'You are using the runtime-only build of Vue where the template ' +
            'compiler is not available. Either pre-compile the templates into ' +
            'render functions, or use the compiler-included build.',
            vm
          );
        } else {
          warn(
            'Failed to mount component: template or render function not defined.',
            vm
          );
        }
      }
    }
    callHook(vm, 'beforeMount');

    var updateComponent;
    /* istanbul ignore if */
    if (config.performance && mark) {
      updateComponent = function () {
        var name = vm._name;
        var id = vm._uid;
        var startTag = "vue-perf-start:" + id;
        var endTag = "vue-perf-end:" + id;

        mark(startTag);
        var vnode = vm._render();
        mark(endTag);
        measure(("vue " + name + " render"), startTag, endTag);

        mark(startTag);
        vm._update(vnode, hydrating);
        mark(endTag);
        measure(("vue " + name + " patch"), startTag, endTag);
      };
    } else {
      updateComponent = function () {
        vm._update(vm._render(), hydrating);
      };
    }

    // we set this to vm._watcher inside the watcher's constructor
    // since the watcher's initial patch may call $forceUpdate (e.g. inside child
    // component's mounted hook), which relies on vm._watcher being already defined
    new Watcher(vm, updateComponent, noop, {
      before: function before () {
        if (vm._isMounted && !vm._isDestroyed) {
          callHook(vm, 'beforeUpdate');
        }
      }
    }, true /* isRenderWatcher */);
    hydrating = false;

    // manually mounted instance, call mounted on self
    // mounted is called for render-created child components in its inserted hook
    if (vm.$vnode == null) {
      vm._isMounted = true;
      callHook(vm, 'mounted');
    }
    return vm
  }

  function updateChildComponent (
    vm,
    propsData,
    listeners,
    parentVnode,
    renderChildren
  ) {
    {
      isUpdatingChildComponent = true;
    }

    // determine whether component has slot children
    // we need to do this before overwriting $options._renderChildren.

    // check if there are dynamic scopedSlots (hand-written or compiled but with
    // dynamic slot names). Static scoped slots compiled from template has the
    // "$stable" marker.
    var newScopedSlots = parentVnode.data.scopedSlots;
    var oldScopedSlots = vm.$scopedSlots;
    var hasDynamicScopedSlot = !!(
      (newScopedSlots && !newScopedSlots.$stable) ||
      (oldScopedSlots !== emptyObject && !oldScopedSlots.$stable) ||
      (newScopedSlots && vm.$scopedSlots.$key !== newScopedSlots.$key)
    );

    // Any static slot children from the parent may have changed during parent's
    // update. Dynamic scoped slots may also have changed. In such cases, a forced
    // update is necessary to ensure correctness.
    var needsForceUpdate = !!(
      renderChildren ||               // has new static slots
      vm.$options._renderChildren ||  // has old static slots
      hasDynamicScopedSlot
    );

    vm.$options._parentVnode = parentVnode;
    vm.$vnode = parentVnode; // update vm's placeholder node without re-render

    if (vm._vnode) { // update child tree's parent
      vm._vnode.parent = parentVnode;
    }
    vm.$options._renderChildren = renderChildren;

    // update $attrs and $listeners hash
    // these are also reactive so they may trigger child update if the child
    // used them during render
    vm.$attrs = parentVnode.data.attrs || emptyObject;
    vm.$listeners = listeners || emptyObject;

    // update props
    if (propsData && vm.$options.props) {
      toggleObserving(false);
      var props = vm._props;
      var propKeys = vm.$options._propKeys || [];
      for (var i = 0; i < propKeys.length; i++) {
        var key = propKeys[i];
        var propOptions = vm.$options.props; // wtf flow?
        props[key] = validateProp(key, propOptions, propsData, vm);
      }
      toggleObserving(true);
      // keep a copy of raw propsData
      vm.$options.propsData = propsData;
    }

    // update listeners
    listeners = listeners || emptyObject;
    var oldListeners = vm.$options._parentListeners;
    vm.$options._parentListeners = listeners;
    updateComponentListeners(vm, listeners, oldListeners);

    // resolve slots + force update if has children
    if (needsForceUpdate) {
      vm.$slots = resolveSlots(renderChildren, parentVnode.context);
      vm.$forceUpdate();
    }

    {
      isUpdatingChildComponent = false;
    }
  }

  function isInInactiveTree (vm) {
    while (vm && (vm = vm.$parent)) {
      if (vm._inactive) { return true }
    }
    return false
  }

  function activateChildComponent (vm, direct) {
    if (direct) {
      vm._directInactive = false;
      if (isInInactiveTree(vm)) {
        return
      }
    } else if (vm._directInactive) {
      return
    }
    if (vm._inactive || vm._inactive === null) {
      vm._inactive = false;
      for (var i = 0; i < vm.$children.length; i++) {
        activateChildComponent(vm.$children[i]);
      }
      callHook(vm, 'activated');
    }
  }

  function deactivateChildComponent (vm, direct) {
    if (direct) {
      vm._directInactive = true;
      if (isInInactiveTree(vm)) {
        return
      }
    }
    if (!vm._inactive) {
      vm._inactive = true;
      for (var i = 0; i < vm.$children.length; i++) {
        deactivateChildComponent(vm.$children[i]);
      }
      callHook(vm, 'deactivated');
    }
  }

  function callHook (vm, hook) {
    // #7573 disable dep collection when invoking lifecycle hooks
    pushTarget();
    var handlers = vm.$options[hook];
    var info = hook + " hook";
    if (handlers) {
      for (var i = 0, j = handlers.length; i < j; i++) {
        invokeWithErrorHandling(handlers[i], vm, null, vm, info);
      }
    }
    if (vm._hasHookEvent) {
      vm.$emit('hook:' + hook);
    }
    popTarget();
  }

  /*  */

  var MAX_UPDATE_COUNT = 100;

  var queue = [];
  var activatedChildren = [];
  var has = {};
  var circular = {};
  var waiting = false;
  var flushing = false;
  var index = 0;

  /**
   * Reset the scheduler's state.
   */
  function resetSchedulerState () {
    index = queue.length = activatedChildren.length = 0;
    has = {};
    {
      circular = {};
    }
    waiting = flushing = false;
  }

  // Async edge case #6566 requires saving the timestamp when event listeners are
  // attached. However, calling performance.now() has a perf overhead especially
  // if the page has thousands of event listeners. Instead, we take a timestamp
  // every time the scheduler flushes and use that for all event listeners
  // attached during that flush.
  var currentFlushTimestamp = 0;

  // Async edge case fix requires storing an event listener's attach timestamp.
  var getNow = Date.now;

  // Determine what event timestamp the browser is using. Annoyingly, the
  // timestamp can either be hi-res (relative to page load) or low-res
  // (relative to UNIX epoch), so in order to compare time we have to use the
  // same timestamp type when saving the flush timestamp.
  // All IE versions use low-res event timestamps, and have problematic clock
  // implementations (#9632)
  if (inBrowser && !isIE) {
    var performance = window.performance;
    if (
      performance &&
      typeof performance.now === 'function' &&
      getNow() > document.createEvent('Event').timeStamp
    ) {
      // if the event timestamp, although evaluated AFTER the Date.now(), is
      // smaller than it, it means the event is using a hi-res timestamp,
      // and we need to use the hi-res version for event listener timestamps as
      // well.
      getNow = function () { return performance.now(); };
    }
  }

  /**
   * Flush both queues and run the watchers.
   */
  function flushSchedulerQueue () {
    currentFlushTimestamp = getNow();
    flushing = true;
    var watcher, id;

    // Sort queue before flush.
    // This ensures that:
    // 1. Components are updated from parent to child. (because parent is always
    //    created before the child)
    // 2. A component's user watchers are run before its render watcher (because
    //    user watchers are created before the render watcher)
    // 3. If a component is destroyed during a parent component's watcher run,
    //    its watchers can be skipped.
    queue.sort(function (a, b) { return a.id - b.id; });

    // do not cache length because more watchers might be pushed
    // as we run existing watchers
    for (index = 0; index < queue.length; index++) {
      watcher = queue[index];
      if (watcher.before) {
        watcher.before();
      }
      id = watcher.id;
      has[id] = null;
      watcher.run();
      // in dev build, check and stop circular updates.
      if (has[id] != null) {
        circular[id] = (circular[id] || 0) + 1;
        if (circular[id] > MAX_UPDATE_COUNT) {
          warn(
            'You may have an infinite update loop ' + (
              watcher.user
                ? ("in watcher with expression \"" + (watcher.expression) + "\"")
                : "in a component render function."
            ),
            watcher.vm
          );
          break
        }
      }
    }

    // keep copies of post queues before resetting state
    var activatedQueue = activatedChildren.slice();
    var updatedQueue = queue.slice();

    resetSchedulerState();

    // call component updated and activated hooks
    callActivatedHooks(activatedQueue);
    callUpdatedHooks(updatedQueue);

    // devtool hook
    /* istanbul ignore if */
    if (devtools && config.devtools) {
      devtools.emit('flush');
    }
  }

  function callUpdatedHooks (queue) {
    var i = queue.length;
    while (i--) {
      var watcher = queue[i];
      var vm = watcher.vm;
      if (vm._watcher === watcher && vm._isMounted && !vm._isDestroyed) {
        callHook(vm, 'updated');
      }
    }
  }

  /**
   * Queue a kept-alive component that was activated during patch.
   * The queue will be processed after the entire tree has been patched.
   */
  function queueActivatedComponent (vm) {
    // setting _inactive to false here so that a render function can
    // rely on checking whether it's in an inactive tree (e.g. router-view)
    vm._inactive = false;
    activatedChildren.push(vm);
  }

  function callActivatedHooks (queue) {
    for (var i = 0; i < queue.length; i++) {
      queue[i]._inactive = true;
      activateChildComponent(queue[i], true /* true */);
    }
  }

  /**
   * Push a watcher into the watcher queue.
   * Jobs with duplicate IDs will be skipped unless it's
   * pushed when the queue is being flushed.
   */
  function queueWatcher (watcher) {
    var id = watcher.id;
    if (has[id] == null) {
      has[id] = true;
      if (!flushing) {
        queue.push(watcher);
      } else {
        // if already flushing, splice the watcher based on its id
        // if already past its id, it will be run next immediately.
        var i = queue.length - 1;
        while (i > index && queue[i].id > watcher.id) {
          i--;
        }
        queue.splice(i + 1, 0, watcher);
      }
      // queue the flush
      if (!waiting) {
        waiting = true;

        if (!config.async) {
          flushSchedulerQueue();
          return
        }
        nextTick(flushSchedulerQueue);
      }
    }
  }

  /*  */



  var uid$2 = 0;

  /**
   * A watcher parses an expression, collects dependencies,
   * and fires callback when the expression value changes.
   * This is used for both the $watch() api and directives.
   */
  var Watcher = function Watcher (
    vm,
    expOrFn,
    cb,
    options,
    isRenderWatcher
  ) {
    this.vm = vm;
    if (isRenderWatcher) {
      vm._watcher = this;
    }
    vm._watchers.push(this);
    // options
    if (options) {
      this.deep = !!options.deep;
      this.user = !!options.user;
      this.lazy = !!options.lazy;
      this.sync = !!options.sync;
      this.before = options.before;
    } else {
      this.deep = this.user = this.lazy = this.sync = false;
    }
    this.cb = cb;
    this.id = ++uid$2; // uid for batching
    this.active = true;
    this.dirty = this.lazy; // for lazy watchers
    this.deps = [];
    this.newDeps = [];
    this.depIds = new _Set();
    this.newDepIds = new _Set();
    this.expression = expOrFn.toString();
    // parse expression for getter
    if (typeof expOrFn === 'function') {
      this.getter = expOrFn;
    } else {
      this.getter = parsePath(expOrFn);
      if (!this.getter) {
        this.getter = noop;
        warn(
          "Failed watching path: \"" + expOrFn + "\" " +
          'Watcher only accepts simple dot-delimited paths. ' +
          'For full control, use a function instead.',
          vm
        );
      }
    }
    this.value = this.lazy
      ? undefined
      : this.get();
  };

  /**
   * Evaluate the getter, and re-collect dependencies.
   */
  Watcher.prototype.get = function get () {
    pushTarget(this);
    var value;
    var vm = this.vm;
    try {
      value = this.getter.call(vm, vm);
    } catch (e) {
      if (this.user) {
        handleError(e, vm, ("getter for watcher \"" + (this.expression) + "\""));
      } else {
        throw e
      }
    } finally {
      // "touch" every property so they are all tracked as
      // dependencies for deep watching
      if (this.deep) {
        traverse(value);
      }
      popTarget();
      this.cleanupDeps();
    }
    return value
  };

  /**
   * Add a dependency to this directive.
   */
  Watcher.prototype.addDep = function addDep (dep) {
    var id = dep.id;
    if (!this.newDepIds.has(id)) {
      this.newDepIds.add(id);
      this.newDeps.push(dep);
      if (!this.depIds.has(id)) {
        dep.addSub(this);
      }
    }
  };

  /**
   * Clean up for dependency collection.
   */
  Watcher.prototype.cleanupDeps = function cleanupDeps () {
    var i = this.deps.length;
    while (i--) {
      var dep = this.deps[i];
      if (!this.newDepIds.has(dep.id)) {
        dep.removeSub(this);
      }
    }
    var tmp = this.depIds;
    this.depIds = this.newDepIds;
    this.newDepIds = tmp;
    this.newDepIds.clear();
    tmp = this.deps;
    this.deps = this.newDeps;
    this.newDeps = tmp;
    this.newDeps.length = 0;
  };

  /**
   * Subscriber interface.
   * Will be called when a dependency changes.
   */
  Watcher.prototype.update = function update () {
    /* istanbul ignore else */
    if (this.lazy) {
      this.dirty = true;
    } else if (this.sync) {
      this.run();
    } else {
      queueWatcher(this);
    }
  };

  /**
   * Scheduler job interface.
   * Will be called by the scheduler.
   */
  Watcher.prototype.run = function run () {
    if (this.active) {
      var value = this.get();
      if (
        value !== this.value ||
        // Deep watchers and watchers on Object/Arrays should fire even
        // when the value is the same, because the value may
        // have mutated.
        isObject(value) ||
        this.deep
      ) {
        // set new value
        var oldValue = this.value;
        this.value = value;
        if (this.user) {
          try {
            this.cb.call(this.vm, value, oldValue);
          } catch (e) {
            handleError(e, this.vm, ("callback for watcher \"" + (this.expression) + "\""));
          }
        } else {
          this.cb.call(this.vm, value, oldValue);
        }
      }
    }
  };

  /**
   * Evaluate the value of the watcher.
   * This only gets called for lazy watchers.
   */
  Watcher.prototype.evaluate = function evaluate () {
    this.value = this.get();
    this.dirty = false;
  };

  /**
   * Depend on all deps collected by this watcher.
   */
  Watcher.prototype.depend = function depend () {
    var i = this.deps.length;
    while (i--) {
      this.deps[i].depend();
    }
  };

  /**
   * Remove self from all dependencies' subscriber list.
   */
  Watcher.prototype.teardown = function teardown () {
    if (this.active) {
      // remove self from vm's watcher list
      // this is a somewhat expensive operation so we skip it
      // if the vm is being destroyed.
      if (!this.vm._isBeingDestroyed) {
        remove(this.vm._watchers, this);
      }
      var i = this.deps.length;
      while (i--) {
        this.deps[i].removeSub(this);
      }
      this.active = false;
    }
  };

  /*  */

  var sharedPropertyDefinition = {
    enumerable: true,
    configurable: true,
    get: noop,
    set: noop
  };

  function proxy (target, sourceKey, key) {
    sharedPropertyDefinition.get = function proxyGetter () {
      return this[sourceKey][key]
    };
    sharedPropertyDefinition.set = function proxySetter (val) {
      this[sourceKey][key] = val;
    };
    Object.defineProperty(target, key, sharedPropertyDefinition);
  }

  function initState (vm) {
    vm._watchers = [];
    var opts = vm.$options;
    if (opts.props) { initProps(vm, opts.props); }
    if (opts.methods) { initMethods(vm, opts.methods); }
    if (opts.data) {
      initData(vm);
    } else {
      observe(vm._data = {}, true /* asRootData */);
    }
    if (opts.computed) { initComputed(vm, opts.computed); }
    if (opts.watch && opts.watch !== nativeWatch) {
      initWatch(vm, opts.watch);
    }
  }

  function initProps (vm, propsOptions) {
    var propsData = vm.$options.propsData || {};
    var props = vm._props = {};
    // cache prop keys so that future props updates can iterate using Array
    // instead of dynamic object key enumeration.
    var keys = vm.$options._propKeys = [];
    var isRoot = !vm.$parent;
    // root instance props should be converted
    if (!isRoot) {
      toggleObserving(false);
    }
    var loop = function ( key ) {
      keys.push(key);
      var value = validateProp(key, propsOptions, propsData, vm);
      /* istanbul ignore else */
      {
        var hyphenatedKey = hyphenate(key);
        if (isReservedAttribute(hyphenatedKey) ||
            config.isReservedAttr(hyphenatedKey)) {
          warn(
            ("\"" + hyphenatedKey + "\" is a reserved attribute and cannot be used as component prop."),
            vm
          );
        }
        defineReactive$$1(props, key, value, function () {
          if (!isRoot && !isUpdatingChildComponent) {
            warn(
              "Avoid mutating a prop directly since the value will be " +
              "overwritten whenever the parent component re-renders. " +
              "Instead, use a data or computed property based on the prop's " +
              "value. Prop being mutated: \"" + key + "\"",
              vm
            );
          }
        });
      }
      // static props are already proxied on the component's prototype
      // during Vue.extend(). We only need to proxy props defined at
      // instantiation here.
      if (!(key in vm)) {
        proxy(vm, "_props", key);
      }
    };

    for (var key in propsOptions) loop( key );
    toggleObserving(true);
  }

  function initData (vm) {
    var data = vm.$options.data;
    data = vm._data = typeof data === 'function'
      ? getData(data, vm)
      : data || {};
    if (!isPlainObject(data)) {
      data = {};
      warn(
        'data functions should return an object:\n' +
        'https://vuejs.org/v2/guide/components.html#data-Must-Be-a-Function',
        vm
      );
    }
    // proxy data on instance
    var keys = Object.keys(data);
    var props = vm.$options.props;
    var methods = vm.$options.methods;
    var i = keys.length;
    while (i--) {
      var key = keys[i];
      {
        if (methods && hasOwn(methods, key)) {
          warn(
            ("Method \"" + key + "\" has already been defined as a data property."),
            vm
          );
        }
      }
      if (props && hasOwn(props, key)) {
        warn(
          "The data property \"" + key + "\" is already declared as a prop. " +
          "Use prop default value instead.",
          vm
        );
      } else if (!isReserved(key)) {
        proxy(vm, "_data", key);
      }
    }
    // observe data
    observe(data, true /* asRootData */);
  }

  function getData (data, vm) {
    // #7573 disable dep collection when invoking data getters
    pushTarget();
    try {
      return data.call(vm, vm)
    } catch (e) {
      handleError(e, vm, "data()");
      return {}
    } finally {
      popTarget();
    }
  }

  var computedWatcherOptions = { lazy: true };

  function initComputed (vm, computed) {
    // $flow-disable-line
    var watchers = vm._computedWatchers = Object.create(null);
    // computed properties are just getters during SSR
    var isSSR = isServerRendering();

    for (var key in computed) {
      var userDef = computed[key];
      var getter = typeof userDef === 'function' ? userDef : userDef.get;
      if (getter == null) {
        warn(
          ("Getter is missing for computed property \"" + key + "\"."),
          vm
        );
      }

      if (!isSSR) {
        // create internal watcher for the computed property.
        watchers[key] = new Watcher(
          vm,
          getter || noop,
          noop,
          computedWatcherOptions
        );
      }

      // component-defined computed properties are already defined on the
      // component prototype. We only need to define computed properties defined
      // at instantiation here.
      if (!(key in vm)) {
        defineComputed(vm, key, userDef);
      } else {
        if (key in vm.$data) {
          warn(("The computed property \"" + key + "\" is already defined in data."), vm);
        } else if (vm.$options.props && key in vm.$options.props) {
          warn(("The computed property \"" + key + "\" is already defined as a prop."), vm);
        }
      }
    }
  }

  function defineComputed (
    target,
    key,
    userDef
  ) {
    var shouldCache = !isServerRendering();
    if (typeof userDef === 'function') {
      sharedPropertyDefinition.get = shouldCache
        ? createComputedGetter(key)
        : createGetterInvoker(userDef);
      sharedPropertyDefinition.set = noop;
    } else {
      sharedPropertyDefinition.get = userDef.get
        ? shouldCache && userDef.cache !== false
          ? createComputedGetter(key)
          : createGetterInvoker(userDef.get)
        : noop;
      sharedPropertyDefinition.set = userDef.set || noop;
    }
    if (sharedPropertyDefinition.set === noop) {
      sharedPropertyDefinition.set = function () {
        warn(
          ("Computed property \"" + key + "\" was assigned to but it has no setter."),
          this
        );
      };
    }
    Object.defineProperty(target, key, sharedPropertyDefinition);
  }

  function createComputedGetter (key) {
    return function computedGetter () {
      var watcher = this._computedWatchers && this._computedWatchers[key];
      if (watcher) {
        if (watcher.dirty) {
          watcher.evaluate();
        }
        if (Dep.target) {
          watcher.depend();
        }
        return watcher.value
      }
    }
  }

  function createGetterInvoker(fn) {
    return function computedGetter () {
      return fn.call(this, this)
    }
  }

  function initMethods (vm, methods) {
    var props = vm.$options.props;
    for (var key in methods) {
      {
        if (typeof methods[key] !== 'function') {
          warn(
            "Method \"" + key + "\" has type \"" + (typeof methods[key]) + "\" in the component definition. " +
            "Did you reference the function correctly?",
            vm
          );
        }
        if (props && hasOwn(props, key)) {
          warn(
            ("Method \"" + key + "\" has already been defined as a prop."),
            vm
          );
        }
        if ((key in vm) && isReserved(key)) {
          warn(
            "Method \"" + key + "\" conflicts with an existing Vue instance method. " +
            "Avoid defining component methods that start with _ or $."
          );
        }
      }
      vm[key] = typeof methods[key] !== 'function' ? noop : bind(methods[key], vm);
    }
  }

  function initWatch (vm, watch) {
    for (var key in watch) {
      var handler = watch[key];
      if (Array.isArray(handler)) {
        for (var i = 0; i < handler.length; i++) {
          createWatcher(vm, key, handler[i]);
        }
      } else {
        createWatcher(vm, key, handler);
      }
    }
  }

  function createWatcher (
    vm,
    expOrFn,
    handler,
    options
  ) {
    if (isPlainObject(handler)) {
      options = handler;
      handler = handler.handler;
    }
    if (typeof handler === 'string') {
      handler = vm[handler];
    }
    return vm.$watch(expOrFn, handler, options)
  }

  function stateMixin (Vue) {
    // flow somehow has problems with directly declared definition object
    // when using Object.defineProperty, so we have to procedurally build up
    // the object here.
    var dataDef = {};
    dataDef.get = function () { return this._data };
    var propsDef = {};
    propsDef.get = function () { return this._props };
    {
      dataDef.set = function () {
        warn(
          'Avoid replacing instance root $data. ' +
          'Use nested data properties instead.',
          this
        );
      };
      propsDef.set = function () {
        warn("$props is readonly.", this);
      };
    }
    Object.defineProperty(Vue.prototype, '$data', dataDef);
    Object.defineProperty(Vue.prototype, '$props', propsDef);

    Vue.prototype.$set = set;
    Vue.prototype.$delete = del;

    Vue.prototype.$watch = function (
      expOrFn,
      cb,
      options
    ) {
      var vm = this;
      if (isPlainObject(cb)) {
        return createWatcher(vm, expOrFn, cb, options)
      }
      options = options || {};
      options.user = true;
      var watcher = new Watcher(vm, expOrFn, cb, options);
      if (options.immediate) {
        try {
          cb.call(vm, watcher.value);
        } catch (error) {
          handleError(error, vm, ("callback for immediate watcher \"" + (watcher.expression) + "\""));
        }
      }
      return function unwatchFn () {
        watcher.teardown();
      }
    };
  }

  /*  */

  var uid$3 = 0;

  function initMixin (Vue) {
    Vue.prototype._init = function (options) {
      var vm = this;
      // a uid
      vm._uid = uid$3++;

      var startTag, endTag;
      /* istanbul ignore if */
      if (config.performance && mark) {
        startTag = "vue-perf-start:" + (vm._uid);
        endTag = "vue-perf-end:" + (vm._uid);
        mark(startTag);
      }

      // a flag to avoid this being observed
      vm._isVue = true;
      // merge options
      if (options && options._isComponent) {
        // optimize internal component instantiation
        // since dynamic options merging is pretty slow, and none of the
        // internal component options needs special treatment.
        initInternalComponent(vm, options);
      } else {
        vm.$options = mergeOptions(
          resolveConstructorOptions(vm.constructor),
          options || {},
          vm
        );
      }
      /* istanbul ignore else */
      {
        initProxy(vm);
      }
      // expose real self
      vm._self = vm;
      initLifecycle(vm);
      initEvents(vm);
      initRender(vm);
      callHook(vm, 'beforeCreate');
      initInjections(vm); // resolve injections before data/props
      initState(vm);
      initProvide(vm); // resolve provide after data/props
      callHook(vm, 'created');

      /* istanbul ignore if */
      if (config.performance && mark) {
        vm._name = formatComponentName(vm, false);
        mark(endTag);
        measure(("vue " + (vm._name) + " init"), startTag, endTag);
      }

      if (vm.$options.el) {
        vm.$mount(vm.$options.el);
      }
    };
  }

  function initInternalComponent (vm, options) {
    var opts = vm.$options = Object.create(vm.constructor.options);
    // doing this because it's faster than dynamic enumeration.
    var parentVnode = options._parentVnode;
    opts.parent = options.parent;
    opts._parentVnode = parentVnode;

    var vnodeComponentOptions = parentVnode.componentOptions;
    opts.propsData = vnodeComponentOptions.propsData;
    opts._parentListeners = vnodeComponentOptions.listeners;
    opts._renderChildren = vnodeComponentOptions.children;
    opts._componentTag = vnodeComponentOptions.tag;

    if (options.render) {
      opts.render = options.render;
      opts.staticRenderFns = options.staticRenderFns;
    }
  }

  function resolveConstructorOptions (Ctor) {
    var options = Ctor.options;
    if (Ctor.super) {
      var superOptions = resolveConstructorOptions(Ctor.super);
      var cachedSuperOptions = Ctor.superOptions;
      if (superOptions !== cachedSuperOptions) {
        // super option changed,
        // need to resolve new options.
        Ctor.superOptions = superOptions;
        // check if there are any late-modified/attached options (#4976)
        var modifiedOptions = resolveModifiedOptions(Ctor);
        // update base extend options
        if (modifiedOptions) {
          extend(Ctor.extendOptions, modifiedOptions);
        }
        options = Ctor.options = mergeOptions(superOptions, Ctor.extendOptions);
        if (options.name) {
          options.components[options.name] = Ctor;
        }
      }
    }
    return options
  }

  function resolveModifiedOptions (Ctor) {
    var modified;
    var latest = Ctor.options;
    var sealed = Ctor.sealedOptions;
    for (var key in latest) {
      if (latest[key] !== sealed[key]) {
        if (!modified) { modified = {}; }
        modified[key] = latest[key];
      }
    }
    return modified
  }

  function Vue (options) {
    if (!(this instanceof Vue)
    ) {
      warn('Vue is a constructor and should be called with the `new` keyword');
    }
    this._init(options);
  }

  initMixin(Vue);
  stateMixin(Vue);
  eventsMixin(Vue);
  lifecycleMixin(Vue);
  renderMixin(Vue);

  /*  */

  function initUse (Vue) {
    Vue.use = function (plugin) {
      var installedPlugins = (this._installedPlugins || (this._installedPlugins = []));
      if (installedPlugins.indexOf(plugin) > -1) {
        return this
      }

      // additional parameters
      var args = toArray(arguments, 1);
      args.unshift(this);
      if (typeof plugin.install === 'function') {
        plugin.install.apply(plugin, args);
      } else if (typeof plugin === 'function') {
        plugin.apply(null, args);
      }
      installedPlugins.push(plugin);
      return this
    };
  }

  /*  */

  function initMixin$1 (Vue) {
    Vue.mixin = function (mixin) {
      this.options = mergeOptions(this.options, mixin);
      return this
    };
  }

  /*  */

  function initExtend (Vue) {
    /**
     * Each instance constructor, including Vue, has a unique
     * cid. This enables us to create wrapped "child
     * constructors" for prototypal inheritance and cache them.
     */
    Vue.cid = 0;
    var cid = 1;

    /**
     * Class inheritance
     */
    Vue.extend = function (extendOptions) {
      extendOptions = extendOptions || {};
      var Super = this;
      var SuperId = Super.cid;
      var cachedCtors = extendOptions._Ctor || (extendOptions._Ctor = {});
      if (cachedCtors[SuperId]) {
        return cachedCtors[SuperId]
      }

      var name = extendOptions.name || Super.options.name;
      if (name) {
        validateComponentName(name);
      }

      var Sub = function VueComponent (options) {
        this._init(options);
      };
      Sub.prototype = Object.create(Super.prototype);
      Sub.prototype.constructor = Sub;
      Sub.cid = cid++;
      Sub.options = mergeOptions(
        Super.options,
        extendOptions
      );
      Sub['super'] = Super;

      // For props and computed properties, we define the proxy getters on
      // the Vue instances at extension time, on the extended prototype. This
      // avoids Object.defineProperty calls for each instance created.
      if (Sub.options.props) {
        initProps$1(Sub);
      }
      if (Sub.options.computed) {
        initComputed$1(Sub);
      }

      // allow further extension/mixin/plugin usage
      Sub.extend = Super.extend;
      Sub.mixin = Super.mixin;
      Sub.use = Super.use;

      // create asset registers, so extended classes
      // can have their private assets too.
      ASSET_TYPES.forEach(function (type) {
        Sub[type] = Super[type];
      });
      // enable recursive self-lookup
      if (name) {
        Sub.options.components[name] = Sub;
      }

      // keep a reference to the super options at extension time.
      // later at instantiation we can check if Super's options have
      // been updated.
      Sub.superOptions = Super.options;
      Sub.extendOptions = extendOptions;
      Sub.sealedOptions = extend({}, Sub.options);

      // cache constructor
      cachedCtors[SuperId] = Sub;
      return Sub
    };
  }

  function initProps$1 (Comp) {
    var props = Comp.options.props;
    for (var key in props) {
      proxy(Comp.prototype, "_props", key);
    }
  }

  function initComputed$1 (Comp) {
    var computed = Comp.options.computed;
    for (var key in computed) {
      defineComputed(Comp.prototype, key, computed[key]);
    }
  }

  /*  */

  function initAssetRegisters (Vue) {
    /**
     * Create asset registration methods.
     */
    ASSET_TYPES.forEach(function (type) {
      Vue[type] = function (
        id,
        definition
      ) {
        if (!definition) {
          return this.options[type + 's'][id]
        } else {
          /* istanbul ignore if */
          if (type === 'component') {
            validateComponentName(id);
          }
          if (type === 'component' && isPlainObject(definition)) {
            definition.name = definition.name || id;
            definition = this.options._base.extend(definition);
          }
          if (type === 'directive' && typeof definition === 'function') {
            definition = { bind: definition, update: definition };
          }
          this.options[type + 's'][id] = definition;
          return definition
        }
      };
    });
  }

  /*  */



  function getComponentName (opts) {
    return opts && (opts.Ctor.options.name || opts.tag)
  }

  function matches (pattern, name) {
    if (Array.isArray(pattern)) {
      return pattern.indexOf(name) > -1
    } else if (typeof pattern === 'string') {
      return pattern.split(',').indexOf(name) > -1
    } else if (isRegExp(pattern)) {
      return pattern.test(name)
    }
    /* istanbul ignore next */
    return false
  }

  function pruneCache (keepAliveInstance, filter) {
    var cache = keepAliveInstance.cache;
    var keys = keepAliveInstance.keys;
    var _vnode = keepAliveInstance._vnode;
    for (var key in cache) {
      var cachedNode = cache[key];
      if (cachedNode) {
        var name = getComponentName(cachedNode.componentOptions);
        if (name && !filter(name)) {
          pruneCacheEntry(cache, key, keys, _vnode);
        }
      }
    }
  }

  function pruneCacheEntry (
    cache,
    key,
    keys,
    current
  ) {
    var cached$$1 = cache[key];
    if (cached$$1 && (!current || cached$$1.tag !== current.tag)) {
      cached$$1.componentInstance.$destroy();
    }
    cache[key] = null;
    remove(keys, key);
  }

  var patternTypes = [String, RegExp, Array];

  var KeepAlive = {
    name: 'keep-alive',
    abstract: true,

    props: {
      include: patternTypes,
      exclude: patternTypes,
      max: [String, Number]
    },

    created: function created () {
      this.cache = Object.create(null);
      this.keys = [];
    },

    destroyed: function destroyed () {
      for (var key in this.cache) {
        pruneCacheEntry(this.cache, key, this.keys);
      }
    },

    mounted: function mounted () {
      var this$1 = this;

      this.$watch('include', function (val) {
        pruneCache(this$1, function (name) { return matches(val, name); });
      });
      this.$watch('exclude', function (val) {
        pruneCache(this$1, function (name) { return !matches(val, name); });
      });
    },

    render: function render () {
      var slot = this.$slots.default;
      var vnode = getFirstComponentChild(slot);
      var componentOptions = vnode && vnode.componentOptions;
      if (componentOptions) {
        // check pattern
        var name = getComponentName(componentOptions);
        var ref = this;
        var include = ref.include;
        var exclude = ref.exclude;
        if (
          // not included
          (include && (!name || !matches(include, name))) ||
          // excluded
          (exclude && name && matches(exclude, name))
        ) {
          return vnode
        }

        var ref$1 = this;
        var cache = ref$1.cache;
        var keys = ref$1.keys;
        var key = vnode.key == null
          // same constructor may get registered as different local components
          // so cid alone is not enough (#3269)
          ? componentOptions.Ctor.cid + (componentOptions.tag ? ("::" + (componentOptions.tag)) : '')
          : vnode.key;
        if (cache[key]) {
          vnode.componentInstance = cache[key].componentInstance;
          // make current key freshest
          remove(keys, key);
          keys.push(key);
        } else {
          cache[key] = vnode;
          keys.push(key);
          // prune oldest entry
          if (this.max && keys.length > parseInt(this.max)) {
            pruneCacheEntry(cache, keys[0], keys, this._vnode);
          }
        }

        vnode.data.keepAlive = true;
      }
      return vnode || (slot && slot[0])
    }
  };

  var builtInComponents = {
    KeepAlive: KeepAlive
  };

  /*  */

  function initGlobalAPI (Vue) {
    // config
    var configDef = {};
    configDef.get = function () { return config; };
    {
      configDef.set = function () {
        warn(
          'Do not replace the Vue.config object, set individual fields instead.'
        );
      };
    }
    Object.defineProperty(Vue, 'config', configDef);

    // exposed util methods.
    // NOTE: these are not considered part of the public API - avoid relying on
    // them unless you are aware of the risk.
    Vue.util = {
      warn: warn,
      extend: extend,
      mergeOptions: mergeOptions,
      defineReactive: defineReactive$$1
    };

    Vue.set = set;
    Vue.delete = del;
    Vue.nextTick = nextTick;

    // 2.6 explicit observable API
    Vue.observable = function (obj) {
      observe(obj);
      return obj
    };

    Vue.options = Object.create(null);
    ASSET_TYPES.forEach(function (type) {
      Vue.options[type + 's'] = Object.create(null);
    });

    // this is used to identify the "base" constructor to extend all plain-object
    // components with in Weex's multi-instance scenarios.
    Vue.options._base = Vue;

    extend(Vue.options.components, builtInComponents);

    initUse(Vue);
    initMixin$1(Vue);
    initExtend(Vue);
    initAssetRegisters(Vue);
  }

  initGlobalAPI(Vue);

  Object.defineProperty(Vue.prototype, '$isServer', {
    get: isServerRendering
  });

  Object.defineProperty(Vue.prototype, '$ssrContext', {
    get: function get () {
      /* istanbul ignore next */
      return this.$vnode && this.$vnode.ssrContext
    }
  });

  // expose FunctionalRenderContext for ssr runtime helper installation
  Object.defineProperty(Vue, 'FunctionalRenderContext', {
    value: FunctionalRenderContext
  });

  Vue.version = '2.6.10';

  /*  */

  // these are reserved for web because they are directly compiled away
  // during template compilation
  var isReservedAttr = makeMap('style,class');

  // attributes that should be using props for binding
  var acceptValue = makeMap('input,textarea,option,select,progress');
  var mustUseProp = function (tag, type, attr) {
    return (
      (attr === 'value' && acceptValue(tag)) && type !== 'button' ||
      (attr === 'selected' && tag === 'option') ||
      (attr === 'checked' && tag === 'input') ||
      (attr === 'muted' && tag === 'video')
    )
  };

  var isEnumeratedAttr = makeMap('contenteditable,draggable,spellcheck');

  var isValidContentEditableValue = makeMap('events,caret,typing,plaintext-only');

  var convertEnumeratedValue = function (key, value) {
    return isFalsyAttrValue(value) || value === 'false'
      ? 'false'
      // allow arbitrary string value for contenteditable
      : key === 'contenteditable' && isValidContentEditableValue(value)
        ? value
        : 'true'
  };

  var isBooleanAttr = makeMap(
    'allowfullscreen,async,autofocus,autoplay,checked,compact,controls,declare,' +
    'default,defaultchecked,defaultmuted,defaultselected,defer,disabled,' +
    'enabled,formnovalidate,hidden,indeterminate,inert,ismap,itemscope,loop,multiple,' +
    'muted,nohref,noresize,noshade,novalidate,nowrap,open,pauseonexit,readonly,' +
    'required,reversed,scoped,seamless,selected,sortable,translate,' +
    'truespeed,typemustmatch,visible'
  );

  var xlinkNS = 'http://www.w3.org/1999/xlink';

  var isXlink = function (name) {
    return name.charAt(5) === ':' && name.slice(0, 5) === 'xlink'
  };

  var getXlinkProp = function (name) {
    return isXlink(name) ? name.slice(6, name.length) : ''
  };

  var isFalsyAttrValue = function (val) {
    return val == null || val === false
  };

  /*  */

  function genClassForVnode (vnode) {
    var data = vnode.data;
    var parentNode = vnode;
    var childNode = vnode;
    while (isDef(childNode.componentInstance)) {
      childNode = childNode.componentInstance._vnode;
      if (childNode && childNode.data) {
        data = mergeClassData(childNode.data, data);
      }
    }
    while (isDef(parentNode = parentNode.parent)) {
      if (parentNode && parentNode.data) {
        data = mergeClassData(data, parentNode.data);
      }
    }
    return renderClass(data.staticClass, data.class)
  }

  function mergeClassData (child, parent) {
    return {
      staticClass: concat(child.staticClass, parent.staticClass),
      class: isDef(child.class)
        ? [child.class, parent.class]
        : parent.class
    }
  }

  function renderClass (
    staticClass,
    dynamicClass
  ) {
    if (isDef(staticClass) || isDef(dynamicClass)) {
      return concat(staticClass, stringifyClass(dynamicClass))
    }
    /* istanbul ignore next */
    return ''
  }

  function concat (a, b) {
    return a ? b ? (a + ' ' + b) : a : (b || '')
  }

  function stringifyClass (value) {
    if (Array.isArray(value)) {
      return stringifyArray(value)
    }
    if (isObject(value)) {
      return stringifyObject(value)
    }
    if (typeof value === 'string') {
      return value
    }
    /* istanbul ignore next */
    return ''
  }

  function stringifyArray (value) {
    var res = '';
    var stringified;
    for (var i = 0, l = value.length; i < l; i++) {
      if (isDef(stringified = stringifyClass(value[i])) && stringified !== '') {
        if (res) { res += ' '; }
        res += stringified;
      }
    }
    return res
  }

  function stringifyObject (value) {
    var res = '';
    for (var key in value) {
      if (value[key]) {
        if (res) { res += ' '; }
        res += key;
      }
    }
    return res
  }

  /*  */

  var namespaceMap = {
    svg: 'http://www.w3.org/2000/svg',
    math: 'http://www.w3.org/1998/Math/MathML'
  };

  var isHTMLTag = makeMap(
    'html,body,base,head,link,meta,style,title,' +
    'address,article,aside,footer,header,h1,h2,h3,h4,h5,h6,hgroup,nav,section,' +
    'div,dd,dl,dt,figcaption,figure,picture,hr,img,li,main,ol,p,pre,ul,' +
    'a,b,abbr,bdi,bdo,br,cite,code,data,dfn,em,i,kbd,mark,q,rp,rt,rtc,ruby,' +
    's,samp,small,span,strong,sub,sup,time,u,var,wbr,area,audio,map,track,video,' +
    'embed,object,param,source,canvas,script,noscript,del,ins,' +
    'caption,col,colgroup,table,thead,tbody,td,th,tr,' +
    'button,datalist,fieldset,form,input,label,legend,meter,optgroup,option,' +
    'output,progress,select,textarea,' +
    'details,dialog,menu,menuitem,summary,' +
    'content,element,shadow,template,blockquote,iframe,tfoot'
  );

  // this map is intentionally selective, only covering SVG elements that may
  // contain child elements.
  var isSVG = makeMap(
    'svg,animate,circle,clippath,cursor,defs,desc,ellipse,filter,font-face,' +
    'foreignObject,g,glyph,image,line,marker,mask,missing-glyph,path,pattern,' +
    'polygon,polyline,rect,switch,symbol,text,textpath,tspan,use,view',
    true
  );

  var isReservedTag = function (tag) {
    return isHTMLTag(tag) || isSVG(tag)
  };

  function getTagNamespace (tag) {
    if (isSVG(tag)) {
      return 'svg'
    }
    // basic support for MathML
    // note it doesn't support other MathML elements being component roots
    if (tag === 'math') {
      return 'math'
    }
  }

  var unknownElementCache = Object.create(null);
  function isUnknownElement (tag) {
    /* istanbul ignore if */
    if (!inBrowser) {
      return true
    }
    if (isReservedTag(tag)) {
      return false
    }
    tag = tag.toLowerCase();
    /* istanbul ignore if */
    if (unknownElementCache[tag] != null) {
      return unknownElementCache[tag]
    }
    var el = document.createElement(tag);
    if (tag.indexOf('-') > -1) {
      // http://stackoverflow.com/a/28210364/1070244
      return (unknownElementCache[tag] = (
        el.constructor === window.HTMLUnknownElement ||
        el.constructor === window.HTMLElement
      ))
    } else {
      return (unknownElementCache[tag] = /HTMLUnknownElement/.test(el.toString()))
    }
  }

  var isTextInputType = makeMap('text,number,password,search,email,tel,url');

  /*  */

  /**
   * Query an element selector if it's not an element already.
   */
  function query (el) {
    if (typeof el === 'string') {
      var selected = document.querySelector(el);
      if (!selected) {
        warn(
          'Cannot find element: ' + el
        );
        return document.createElement('div')
      }
      return selected
    } else {
      return el
    }
  }

  /*  */

  function createElement$1 (tagName, vnode) {
    var elm = document.createElement(tagName);
    if (tagName !== 'select') {
      return elm
    }
    // false or null will remove the attribute but undefined will not
    if (vnode.data && vnode.data.attrs && vnode.data.attrs.multiple !== undefined) {
      elm.setAttribute('multiple', 'multiple');
    }
    return elm
  }

  function createElementNS (namespace, tagName) {
    return document.createElementNS(namespaceMap[namespace], tagName)
  }

  function createTextNode (text) {
    return document.createTextNode(text)
  }

  function createComment (text) {
    return document.createComment(text)
  }

  function insertBefore (parentNode, newNode, referenceNode) {
    parentNode.insertBefore(newNode, referenceNode);
  }

  function removeChild (node, child) {
    node.removeChild(child);
  }

  function appendChild (node, child) {
    node.appendChild(child);
  }

  function parentNode (node) {
    return node.parentNode
  }

  function nextSibling (node) {
    return node.nextSibling
  }

  function tagName (node) {
    return node.tagName
  }

  function setTextContent (node, text) {
    node.textContent = text;
  }

  function setStyleScope (node, scopeId) {
    node.setAttribute(scopeId, '');
  }

  var nodeOps = /*#__PURE__*/Object.freeze({
    createElement: createElement$1,
    createElementNS: createElementNS,
    createTextNode: createTextNode,
    createComment: createComment,
    insertBefore: insertBefore,
    removeChild: removeChild,
    appendChild: appendChild,
    parentNode: parentNode,
    nextSibling: nextSibling,
    tagName: tagName,
    setTextContent: setTextContent,
    setStyleScope: setStyleScope
  });

  /*  */

  var ref = {
    create: function create (_, vnode) {
      registerRef(vnode);
    },
    update: function update (oldVnode, vnode) {
      if (oldVnode.data.ref !== vnode.data.ref) {
        registerRef(oldVnode, true);
        registerRef(vnode);
      }
    },
    destroy: function destroy (vnode) {
      registerRef(vnode, true);
    }
  };

  function registerRef (vnode, isRemoval) {
    var key = vnode.data.ref;
    if (!isDef(key)) { return }

    var vm = vnode.context;
    var ref = vnode.componentInstance || vnode.elm;
    var refs = vm.$refs;
    if (isRemoval) {
      if (Array.isArray(refs[key])) {
        remove(refs[key], ref);
      } else if (refs[key] === ref) {
        refs[key] = undefined;
      }
    } else {
      if (vnode.data.refInFor) {
        if (!Array.isArray(refs[key])) {
          refs[key] = [ref];
        } else if (refs[key].indexOf(ref) < 0) {
          // $flow-disable-line
          refs[key].push(ref);
        }
      } else {
        refs[key] = ref;
      }
    }
  }

  /**
   * Virtual DOM patching algorithm based on Snabbdom by
   * Simon Friis Vindum (@paldepind)
   * Licensed under the MIT License
   * https://github.com/paldepind/snabbdom/blob/master/LICENSE
   *
   * modified by Evan You (@yyx990803)
   *
   * Not type-checking this because this file is perf-critical and the cost
   * of making flow understand it is not worth it.
   */

  var emptyNode = new VNode('', {}, []);

  var hooks = ['create', 'activate', 'update', 'remove', 'destroy'];

  function sameVnode (a, b) {
    return (
      a.key === b.key && (
        (
          a.tag === b.tag &&
          a.isComment === b.isComment &&
          isDef(a.data) === isDef(b.data) &&
          sameInputType(a, b)
        ) || (
          isTrue(a.isAsyncPlaceholder) &&
          a.asyncFactory === b.asyncFactory &&
          isUndef(b.asyncFactory.error)
        )
      )
    )
  }

  function sameInputType (a, b) {
    if (a.tag !== 'input') { return true }
    var i;
    var typeA = isDef(i = a.data) && isDef(i = i.attrs) && i.type;
    var typeB = isDef(i = b.data) && isDef(i = i.attrs) && i.type;
    return typeA === typeB || isTextInputType(typeA) && isTextInputType(typeB)
  }

  function createKeyToOldIdx (children, beginIdx, endIdx) {
    var i, key;
    var map = {};
    for (i = beginIdx; i <= endIdx; ++i) {
      key = children[i].key;
      if (isDef(key)) { map[key] = i; }
    }
    return map
  }

  function createPatchFunction (backend) {
    var i, j;
    var cbs = {};

    var modules = backend.modules;
    var nodeOps = backend.nodeOps;

    for (i = 0; i < hooks.length; ++i) {
      cbs[hooks[i]] = [];
      for (j = 0; j < modules.length; ++j) {
        if (isDef(modules[j][hooks[i]])) {
          cbs[hooks[i]].push(modules[j][hooks[i]]);
        }
      }
    }

    function emptyNodeAt (elm) {
      return new VNode(nodeOps.tagName(elm).toLowerCase(), {}, [], undefined, elm)
    }

    function createRmCb (childElm, listeners) {
      function remove$$1 () {
        if (--remove$$1.listeners === 0) {
          removeNode(childElm);
        }
      }
      remove$$1.listeners = listeners;
      return remove$$1
    }

    function removeNode (el) {
      var parent = nodeOps.parentNode(el);
      // element may have already been removed due to v-html / v-text
      if (isDef(parent)) {
        nodeOps.removeChild(parent, el);
      }
    }

    function isUnknownElement$$1 (vnode, inVPre) {
      return (
        !inVPre &&
        !vnode.ns &&
        !(
          config.ignoredElements.length &&
          config.ignoredElements.some(function (ignore) {
            return isRegExp(ignore)
              ? ignore.test(vnode.tag)
              : ignore === vnode.tag
          })
        ) &&
        config.isUnknownElement(vnode.tag)
      )
    }

    var creatingElmInVPre = 0;

    function createElm (
      vnode,
      insertedVnodeQueue,
      parentElm,
      refElm,
      nested,
      ownerArray,
      index
    ) {
      if (isDef(vnode.elm) && isDef(ownerArray)) {
        // This vnode was used in a previous render!
        // now it's used as a new node, overwriting its elm would cause
        // potential patch errors down the road when it's used as an insertion
        // reference node. Instead, we clone the node on-demand before creating
        // associated DOM element for it.
        vnode = ownerArray[index] = cloneVNode(vnode);
      }

      vnode.isRootInsert = !nested; // for transition enter check
      if (createComponent(vnode, insertedVnodeQueue, parentElm, refElm)) {
        return
      }

      var data = vnode.data;
      var children = vnode.children;
      var tag = vnode.tag;
      if (isDef(tag)) {
        {
          if (data && data.pre) {
            creatingElmInVPre++;
          }
          if (isUnknownElement$$1(vnode, creatingElmInVPre)) {
            warn(
              'Unknown custom element: <' + tag + '> - did you ' +
              'register the component correctly? For recursive components, ' +
              'make sure to provide the "name" option.',
              vnode.context
            );
          }
        }

        vnode.elm = vnode.ns
          ? nodeOps.createElementNS(vnode.ns, tag)
          : nodeOps.createElement(tag, vnode);
        setScope(vnode);

        /* istanbul ignore if */
        {
          createChildren(vnode, children, insertedVnodeQueue);
          if (isDef(data)) {
            invokeCreateHooks(vnode, insertedVnodeQueue);
          }
          insert(parentElm, vnode.elm, refElm);
        }

        if (data && data.pre) {
          creatingElmInVPre--;
        }
      } else if (isTrue(vnode.isComment)) {
        vnode.elm = nodeOps.createComment(vnode.text);
        insert(parentElm, vnode.elm, refElm);
      } else {
        vnode.elm = nodeOps.createTextNode(vnode.text);
        insert(parentElm, vnode.elm, refElm);
      }
    }

    function createComponent (vnode, insertedVnodeQueue, parentElm, refElm) {
      var i = vnode.data;
      if (isDef(i)) {
        var isReactivated = isDef(vnode.componentInstance) && i.keepAlive;
        if (isDef(i = i.hook) && isDef(i = i.init)) {
          i(vnode, false /* hydrating */);
        }
        // after calling the init hook, if the vnode is a child component
        // it should've created a child instance and mounted it. the child
        // component also has set the placeholder vnode's elm.
        // in that case we can just return the element and be done.
        if (isDef(vnode.componentInstance)) {
          initComponent(vnode, insertedVnodeQueue);
          insert(parentElm, vnode.elm, refElm);
          if (isTrue(isReactivated)) {
            reactivateComponent(vnode, insertedVnodeQueue, parentElm, refElm);
          }
          return true
        }
      }
    }

    function initComponent (vnode, insertedVnodeQueue) {
      if (isDef(vnode.data.pendingInsert)) {
        insertedVnodeQueue.push.apply(insertedVnodeQueue, vnode.data.pendingInsert);
        vnode.data.pendingInsert = null;
      }
      vnode.elm = vnode.componentInstance.$el;
      if (isPatchable(vnode)) {
        invokeCreateHooks(vnode, insertedVnodeQueue);
        setScope(vnode);
      } else {
        // empty component root.
        // skip all element-related modules except for ref (#3455)
        registerRef(vnode);
        // make sure to invoke the insert hook
        insertedVnodeQueue.push(vnode);
      }
    }

    function reactivateComponent (vnode, insertedVnodeQueue, parentElm, refElm) {
      var i;
      // hack for #4339: a reactivated component with inner transition
      // does not trigger because the inner node's created hooks are not called
      // again. It's not ideal to involve module-specific logic in here but
      // there doesn't seem to be a better way to do it.
      var innerNode = vnode;
      while (innerNode.componentInstance) {
        innerNode = innerNode.componentInstance._vnode;
        if (isDef(i = innerNode.data) && isDef(i = i.transition)) {
          for (i = 0; i < cbs.activate.length; ++i) {
            cbs.activate[i](emptyNode, innerNode);
          }
          insertedVnodeQueue.push(innerNode);
          break
        }
      }
      // unlike a newly created component,
      // a reactivated keep-alive component doesn't insert itself
      insert(parentElm, vnode.elm, refElm);
    }

    function insert (parent, elm, ref$$1) {
      if (isDef(parent)) {
        if (isDef(ref$$1)) {
          if (nodeOps.parentNode(ref$$1) === parent) {
            nodeOps.insertBefore(parent, elm, ref$$1);
          }
        } else {
          nodeOps.appendChild(parent, elm);
        }
      }
    }

    function createChildren (vnode, children, insertedVnodeQueue) {
      if (Array.isArray(children)) {
        {
          checkDuplicateKeys(children);
        }
        for (var i = 0; i < children.length; ++i) {
          createElm(children[i], insertedVnodeQueue, vnode.elm, null, true, children, i);
        }
      } else if (isPrimitive(vnode.text)) {
        nodeOps.appendChild(vnode.elm, nodeOps.createTextNode(String(vnode.text)));
      }
    }

    function isPatchable (vnode) {
      while (vnode.componentInstance) {
        vnode = vnode.componentInstance._vnode;
      }
      return isDef(vnode.tag)
    }

    function invokeCreateHooks (vnode, insertedVnodeQueue) {
      for (var i$1 = 0; i$1 < cbs.create.length; ++i$1) {
        cbs.create[i$1](emptyNode, vnode);
      }
      i = vnode.data.hook; // Reuse variable
      if (isDef(i)) {
        if (isDef(i.create)) { i.create(emptyNode, vnode); }
        if (isDef(i.insert)) { insertedVnodeQueue.push(vnode); }
      }
    }

    // set scope id attribute for scoped CSS.
    // this is implemented as a special case to avoid the overhead
    // of going through the normal attribute patching process.
    function setScope (vnode) {
      var i;
      if (isDef(i = vnode.fnScopeId)) {
        nodeOps.setStyleScope(vnode.elm, i);
      } else {
        var ancestor = vnode;
        while (ancestor) {
          if (isDef(i = ancestor.context) && isDef(i = i.$options._scopeId)) {
            nodeOps.setStyleScope(vnode.elm, i);
          }
          ancestor = ancestor.parent;
        }
      }
      // for slot content they should also get the scopeId from the host instance.
      if (isDef(i = activeInstance) &&
        i !== vnode.context &&
        i !== vnode.fnContext &&
        isDef(i = i.$options._scopeId)
      ) {
        nodeOps.setStyleScope(vnode.elm, i);
      }
    }

    function addVnodes (parentElm, refElm, vnodes, startIdx, endIdx, insertedVnodeQueue) {
      for (; startIdx <= endIdx; ++startIdx) {
        createElm(vnodes[startIdx], insertedVnodeQueue, parentElm, refElm, false, vnodes, startIdx);
      }
    }

    function invokeDestroyHook (vnode) {
      var i, j;
      var data = vnode.data;
      if (isDef(data)) {
        if (isDef(i = data.hook) && isDef(i = i.destroy)) { i(vnode); }
        for (i = 0; i < cbs.destroy.length; ++i) { cbs.destroy[i](vnode); }
      }
      if (isDef(i = vnode.children)) {
        for (j = 0; j < vnode.children.length; ++j) {
          invokeDestroyHook(vnode.children[j]);
        }
      }
    }

    function removeVnodes (parentElm, vnodes, startIdx, endIdx) {
      for (; startIdx <= endIdx; ++startIdx) {
        var ch = vnodes[startIdx];
        if (isDef(ch)) {
          if (isDef(ch.tag)) {
            removeAndInvokeRemoveHook(ch);
            invokeDestroyHook(ch);
          } else { // Text node
            removeNode(ch.elm);
          }
        }
      }
    }

    function removeAndInvokeRemoveHook (vnode, rm) {
      if (isDef(rm) || isDef(vnode.data)) {
        var i;
        var listeners = cbs.remove.length + 1;
        if (isDef(rm)) {
          // we have a recursively passed down rm callback
          // increase the listeners count
          rm.listeners += listeners;
        } else {
          // directly removing
          rm = createRmCb(vnode.elm, listeners);
        }
        // recursively invoke hooks on child component root node
        if (isDef(i = vnode.componentInstance) && isDef(i = i._vnode) && isDef(i.data)) {
          removeAndInvokeRemoveHook(i, rm);
        }
        for (i = 0; i < cbs.remove.length; ++i) {
          cbs.remove[i](vnode, rm);
        }
        if (isDef(i = vnode.data.hook) && isDef(i = i.remove)) {
          i(vnode, rm);
        } else {
          rm();
        }
      } else {
        removeNode(vnode.elm);
      }
    }

    function updateChildren (parentElm, oldCh, newCh, insertedVnodeQueue, removeOnly) {
      var oldStartIdx = 0;
      var newStartIdx = 0;
      var oldEndIdx = oldCh.length - 1;
      var oldStartVnode = oldCh[0];
      var oldEndVnode = oldCh[oldEndIdx];
      var newEndIdx = newCh.length - 1;
      var newStartVnode = newCh[0];
      var newEndVnode = newCh[newEndIdx];
      var oldKeyToIdx, idxInOld, vnodeToMove, refElm;

      // removeOnly is a special flag used only by <transition-group>
      // to ensure removed elements stay in correct relative positions
      // during leaving transitions
      var canMove = !removeOnly;

      {
        checkDuplicateKeys(newCh);
      }

      while (oldStartIdx <= oldEndIdx && newStartIdx <= newEndIdx) {
        if (isUndef(oldStartVnode)) {
          oldStartVnode = oldCh[++oldStartIdx]; // Vnode has been moved left
        } else if (isUndef(oldEndVnode)) {
          oldEndVnode = oldCh[--oldEndIdx];
        } else if (sameVnode(oldStartVnode, newStartVnode)) {
          patchVnode(oldStartVnode, newStartVnode, insertedVnodeQueue, newCh, newStartIdx);
          oldStartVnode = oldCh[++oldStartIdx];
          newStartVnode = newCh[++newStartIdx];
        } else if (sameVnode(oldEndVnode, newEndVnode)) {
          patchVnode(oldEndVnode, newEndVnode, insertedVnodeQueue, newCh, newEndIdx);
          oldEndVnode = oldCh[--oldEndIdx];
          newEndVnode = newCh[--newEndIdx];
        } else if (sameVnode(oldStartVnode, newEndVnode)) { // Vnode moved right
          patchVnode(oldStartVnode, newEndVnode, insertedVnodeQueue, newCh, newEndIdx);
          canMove && nodeOps.insertBefore(parentElm, oldStartVnode.elm, nodeOps.nextSibling(oldEndVnode.elm));
          oldStartVnode = oldCh[++oldStartIdx];
          newEndVnode = newCh[--newEndIdx];
        } else if (sameVnode(oldEndVnode, newStartVnode)) { // Vnode moved left
          patchVnode(oldEndVnode, newStartVnode, insertedVnodeQueue, newCh, newStartIdx);
          canMove && nodeOps.insertBefore(parentElm, oldEndVnode.elm, oldStartVnode.elm);
          oldEndVnode = oldCh[--oldEndIdx];
          newStartVnode = newCh[++newStartIdx];
        } else {
          if (isUndef(oldKeyToIdx)) { oldKeyToIdx = createKeyToOldIdx(oldCh, oldStartIdx, oldEndIdx); }
          idxInOld = isDef(newStartVnode.key)
            ? oldKeyToIdx[newStartVnode.key]
            : findIdxInOld(newStartVnode, oldCh, oldStartIdx, oldEndIdx);
          if (isUndef(idxInOld)) { // New element
            createElm(newStartVnode, insertedVnodeQueue, parentElm, oldStartVnode.elm, false, newCh, newStartIdx);
          } else {
            vnodeToMove = oldCh[idxInOld];
            if (sameVnode(vnodeToMove, newStartVnode)) {
              patchVnode(vnodeToMove, newStartVnode, insertedVnodeQueue, newCh, newStartIdx);
              oldCh[idxInOld] = undefined;
              canMove && nodeOps.insertBefore(parentElm, vnodeToMove.elm, oldStartVnode.elm);
            } else {
              // same key but different element. treat as new element
              createElm(newStartVnode, insertedVnodeQueue, parentElm, oldStartVnode.elm, false, newCh, newStartIdx);
            }
          }
          newStartVnode = newCh[++newStartIdx];
        }
      }
      if (oldStartIdx > oldEndIdx) {
        refElm = isUndef(newCh[newEndIdx + 1]) ? null : newCh[newEndIdx + 1].elm;
        addVnodes(parentElm, refElm, newCh, newStartIdx, newEndIdx, insertedVnodeQueue);
      } else if (newStartIdx > newEndIdx) {
        removeVnodes(parentElm, oldCh, oldStartIdx, oldEndIdx);
      }
    }

    function checkDuplicateKeys (children) {
      var seenKeys = {};
      for (var i = 0; i < children.length; i++) {
        var vnode = children[i];
        var key = vnode.key;
        if (isDef(key)) {
          if (seenKeys[key]) {
            warn(
              ("Duplicate keys detected: '" + key + "'. This may cause an update error."),
              vnode.context
            );
          } else {
            seenKeys[key] = true;
          }
        }
      }
    }

    function findIdxInOld (node, oldCh, start, end) {
      for (var i = start; i < end; i++) {
        var c = oldCh[i];
        if (isDef(c) && sameVnode(node, c)) { return i }
      }
    }

    function patchVnode (
      oldVnode,
      vnode,
      insertedVnodeQueue,
      ownerArray,
      index,
      removeOnly
    ) {
      if (oldVnode === vnode) {
        return
      }

      if (isDef(vnode.elm) && isDef(ownerArray)) {
        // clone reused vnode
        vnode = ownerArray[index] = cloneVNode(vnode);
      }

      var elm = vnode.elm = oldVnode.elm;

      if (isTrue(oldVnode.isAsyncPlaceholder)) {
        if (isDef(vnode.asyncFactory.resolved)) {
          hydrate(oldVnode.elm, vnode, insertedVnodeQueue);
        } else {
          vnode.isAsyncPlaceholder = true;
        }
        return
      }

      // reuse element for static trees.
      // note we only do this if the vnode is cloned -
      // if the new node is not cloned it means the render functions have been
      // reset by the hot-reload-api and we need to do a proper re-render.
      if (isTrue(vnode.isStatic) &&
        isTrue(oldVnode.isStatic) &&
        vnode.key === oldVnode.key &&
        (isTrue(vnode.isCloned) || isTrue(vnode.isOnce))
      ) {
        vnode.componentInstance = oldVnode.componentInstance;
        return
      }

      var i;
      var data = vnode.data;
      if (isDef(data) && isDef(i = data.hook) && isDef(i = i.prepatch)) {
        i(oldVnode, vnode);
      }

      var oldCh = oldVnode.children;
      var ch = vnode.children;
      if (isDef(data) && isPatchable(vnode)) {
        for (i = 0; i < cbs.update.length; ++i) { cbs.update[i](oldVnode, vnode); }
        if (isDef(i = data.hook) && isDef(i = i.update)) { i(oldVnode, vnode); }
      }
      if (isUndef(vnode.text)) {
        if (isDef(oldCh) && isDef(ch)) {
          if (oldCh !== ch) { updateChildren(elm, oldCh, ch, insertedVnodeQueue, removeOnly); }
        } else if (isDef(ch)) {
          {
            checkDuplicateKeys(ch);
          }
          if (isDef(oldVnode.text)) { nodeOps.setTextContent(elm, ''); }
          addVnodes(elm, null, ch, 0, ch.length - 1, insertedVnodeQueue);
        } else if (isDef(oldCh)) {
          removeVnodes(elm, oldCh, 0, oldCh.length - 1);
        } else if (isDef(oldVnode.text)) {
          nodeOps.setTextContent(elm, '');
        }
      } else if (oldVnode.text !== vnode.text) {
        nodeOps.setTextContent(elm, vnode.text);
      }
      if (isDef(data)) {
        if (isDef(i = data.hook) && isDef(i = i.postpatch)) { i(oldVnode, vnode); }
      }
    }

    function invokeInsertHook (vnode, queue, initial) {
      // delay insert hooks for component root nodes, invoke them after the
      // element is really inserted
      if (isTrue(initial) && isDef(vnode.parent)) {
        vnode.parent.data.pendingInsert = queue;
      } else {
        for (var i = 0; i < queue.length; ++i) {
          queue[i].data.hook.insert(queue[i]);
        }
      }
    }

    var hydrationBailed = false;
    // list of modules that can skip create hook during hydration because they
    // are already rendered on the client or has no need for initialization
    // Note: style is excluded because it relies on initial clone for future
    // deep updates (#7063).
    var isRenderedModule = makeMap('attrs,class,staticClass,staticStyle,key');

    // Note: this is a browser-only function so we can assume elms are DOM nodes.
    function hydrate (elm, vnode, insertedVnodeQueue, inVPre) {
      var i;
      var tag = vnode.tag;
      var data = vnode.data;
      var children = vnode.children;
      inVPre = inVPre || (data && data.pre);
      vnode.elm = elm;

      if (isTrue(vnode.isComment) && isDef(vnode.asyncFactory)) {
        vnode.isAsyncPlaceholder = true;
        return true
      }
      // assert node match
      {
        if (!assertNodeMatch(elm, vnode, inVPre)) {
          return false
        }
      }
      if (isDef(data)) {
        if (isDef(i = data.hook) && isDef(i = i.init)) { i(vnode, true /* hydrating */); }
        if (isDef(i = vnode.componentInstance)) {
          // child component. it should have hydrated its own tree.
          initComponent(vnode, insertedVnodeQueue);
          return true
        }
      }
      if (isDef(tag)) {
        if (isDef(children)) {
          // empty element, allow client to pick up and populate children
          if (!elm.hasChildNodes()) {
            createChildren(vnode, children, insertedVnodeQueue);
          } else {
            // v-html and domProps: innerHTML
            if (isDef(i = data) && isDef(i = i.domProps) && isDef(i = i.innerHTML)) {
              if (i !== elm.innerHTML) {
                /* istanbul ignore if */
                if (typeof console !== 'undefined' &&
                  !hydrationBailed
                ) {
                  hydrationBailed = true;
                  console.warn('Parent: ', elm);
                  console.warn('server innerHTML: ', i);
                  console.warn('client innerHTML: ', elm.innerHTML);
                }
                return false
              }
            } else {
              // iterate and compare children lists
              var childrenMatch = true;
              var childNode = elm.firstChild;
              for (var i$1 = 0; i$1 < children.length; i$1++) {
                if (!childNode || !hydrate(childNode, children[i$1], insertedVnodeQueue, inVPre)) {
                  childrenMatch = false;
                  break
                }
                childNode = childNode.nextSibling;
              }
              // if childNode is not null, it means the actual childNodes list is
              // longer than the virtual children list.
              if (!childrenMatch || childNode) {
                /* istanbul ignore if */
                if (typeof console !== 'undefined' &&
                  !hydrationBailed
                ) {
                  hydrationBailed = true;
                  console.warn('Parent: ', elm);
                  console.warn('Mismatching childNodes vs. VNodes: ', elm.childNodes, children);
                }
                return false
              }
            }
          }
        }
        if (isDef(data)) {
          var fullInvoke = false;
          for (var key in data) {
            if (!isRenderedModule(key)) {
              fullInvoke = true;
              invokeCreateHooks(vnode, insertedVnodeQueue);
              break
            }
          }
          if (!fullInvoke && data['class']) {
            // ensure collecting deps for deep class bindings for future updates
            traverse(data['class']);
          }
        }
      } else if (elm.data !== vnode.text) {
        elm.data = vnode.text;
      }
      return true
    }

    function assertNodeMatch (node, vnode, inVPre) {
      if (isDef(vnode.tag)) {
        return vnode.tag.indexOf('vue-component') === 0 || (
          !isUnknownElement$$1(vnode, inVPre) &&
          vnode.tag.toLowerCase() === (node.tagName && node.tagName.toLowerCase())
        )
      } else {
        return node.nodeType === (vnode.isComment ? 8 : 3)
      }
    }

    return function patch (oldVnode, vnode, hydrating, removeOnly) {
      if (isUndef(vnode)) {
        if (isDef(oldVnode)) { invokeDestroyHook(oldVnode); }
        return
      }

      var isInitialPatch = false;
      var insertedVnodeQueue = [];

      if (isUndef(oldVnode)) {
        // empty mount (likely as component), create new root element
        isInitialPatch = true;
        createElm(vnode, insertedVnodeQueue);
      } else {
        var isRealElement = isDef(oldVnode.nodeType);
        if (!isRealElement && sameVnode(oldVnode, vnode)) {
          // patch existing root node
          patchVnode(oldVnode, vnode, insertedVnodeQueue, null, null, removeOnly);
        } else {
          if (isRealElement) {
            // mounting to a real element
            // check if this is server-rendered content and if we can perform
            // a successful hydration.
            if (oldVnode.nodeType === 1 && oldVnode.hasAttribute(SSR_ATTR)) {
              oldVnode.removeAttribute(SSR_ATTR);
              hydrating = true;
            }
            if (isTrue(hydrating)) {
              if (hydrate(oldVnode, vnode, insertedVnodeQueue)) {
                invokeInsertHook(vnode, insertedVnodeQueue, true);
                return oldVnode
              } else {
                warn(
                  'The client-side rendered virtual DOM tree is not matching ' +
                  'server-rendered content. This is likely caused by incorrect ' +
                  'HTML markup, for example nesting block-level elements inside ' +
                  '<p>, or missing <tbody>. Bailing hydration and performing ' +
                  'full client-side render.'
                );
              }
            }
            // either not server-rendered, or hydration failed.
            // create an empty node and replace it
            oldVnode = emptyNodeAt(oldVnode);
          }

          // replacing existing element
          var oldElm = oldVnode.elm;
          var parentElm = nodeOps.parentNode(oldElm);

          // create new node
          createElm(
            vnode,
            insertedVnodeQueue,
            // extremely rare edge case: do not insert if old element is in a
            // leaving transition. Only happens when combining transition +
            // keep-alive + HOCs. (#4590)
            oldElm._leaveCb ? null : parentElm,
            nodeOps.nextSibling(oldElm)
          );

          // update parent placeholder node element, recursively
          if (isDef(vnode.parent)) {
            var ancestor = vnode.parent;
            var patchable = isPatchable(vnode);
            while (ancestor) {
              for (var i = 0; i < cbs.destroy.length; ++i) {
                cbs.destroy[i](ancestor);
              }
              ancestor.elm = vnode.elm;
              if (patchable) {
                for (var i$1 = 0; i$1 < cbs.create.length; ++i$1) {
                  cbs.create[i$1](emptyNode, ancestor);
                }
                // #6513
                // invoke insert hooks that may have been merged by create hooks.
                // e.g. for directives that uses the "inserted" hook.
                var insert = ancestor.data.hook.insert;
                if (insert.merged) {
                  // start at index 1 to avoid re-invoking component mounted hook
                  for (var i$2 = 1; i$2 < insert.fns.length; i$2++) {
                    insert.fns[i$2]();
                  }
                }
              } else {
                registerRef(ancestor);
              }
              ancestor = ancestor.parent;
            }
          }

          // destroy old node
          if (isDef(parentElm)) {
            removeVnodes(parentElm, [oldVnode], 0, 0);
          } else if (isDef(oldVnode.tag)) {
            invokeDestroyHook(oldVnode);
          }
        }
      }

      invokeInsertHook(vnode, insertedVnodeQueue, isInitialPatch);
      return vnode.elm
    }
  }

  /*  */

  var directives = {
    create: updateDirectives,
    update: updateDirectives,
    destroy: function unbindDirectives (vnode) {
      updateDirectives(vnode, emptyNode);
    }
  };

  function updateDirectives (oldVnode, vnode) {
    if (oldVnode.data.directives || vnode.data.directives) {
      _update(oldVnode, vnode);
    }
  }

  function _update (oldVnode, vnode) {
    var isCreate = oldVnode === emptyNode;
    var isDestroy = vnode === emptyNode;
    var oldDirs = normalizeDirectives$1(oldVnode.data.directives, oldVnode.context);
    var newDirs = normalizeDirectives$1(vnode.data.directives, vnode.context);

    var dirsWithInsert = [];
    var dirsWithPostpatch = [];

    var key, oldDir, dir;
    for (key in newDirs) {
      oldDir = oldDirs[key];
      dir = newDirs[key];
      if (!oldDir) {
        // new directive, bind
        callHook$1(dir, 'bind', vnode, oldVnode);
        if (dir.def && dir.def.inserted) {
          dirsWithInsert.push(dir);
        }
      } else {
        // existing directive, update
        dir.oldValue = oldDir.value;
        dir.oldArg = oldDir.arg;
        callHook$1(dir, 'update', vnode, oldVnode);
        if (dir.def && dir.def.componentUpdated) {
          dirsWithPostpatch.push(dir);
        }
      }
    }

    if (dirsWithInsert.length) {
      var callInsert = function () {
        for (var i = 0; i < dirsWithInsert.length; i++) {
          callHook$1(dirsWithInsert[i], 'inserted', vnode, oldVnode);
        }
      };
      if (isCreate) {
        mergeVNodeHook(vnode, 'insert', callInsert);
      } else {
        callInsert();
      }
    }

    if (dirsWithPostpatch.length) {
      mergeVNodeHook(vnode, 'postpatch', function () {
        for (var i = 0; i < dirsWithPostpatch.length; i++) {
          callHook$1(dirsWithPostpatch[i], 'componentUpdated', vnode, oldVnode);
        }
      });
    }

    if (!isCreate) {
      for (key in oldDirs) {
        if (!newDirs[key]) {
          // no longer present, unbind
          callHook$1(oldDirs[key], 'unbind', oldVnode, oldVnode, isDestroy);
        }
      }
    }
  }

  var emptyModifiers = Object.create(null);

  function normalizeDirectives$1 (
    dirs,
    vm
  ) {
    var res = Object.create(null);
    if (!dirs) {
      // $flow-disable-line
      return res
    }
    var i, dir;
    for (i = 0; i < dirs.length; i++) {
      dir = dirs[i];
      if (!dir.modifiers) {
        // $flow-disable-line
        dir.modifiers = emptyModifiers;
      }
      res[getRawDirName(dir)] = dir;
      dir.def = resolveAsset(vm.$options, 'directives', dir.name, true);
    }
    // $flow-disable-line
    return res
  }

  function getRawDirName (dir) {
    return dir.rawName || ((dir.name) + "." + (Object.keys(dir.modifiers || {}).join('.')))
  }

  function callHook$1 (dir, hook, vnode, oldVnode, isDestroy) {
    var fn = dir.def && dir.def[hook];
    if (fn) {
      try {
        fn(vnode.elm, dir, vnode, oldVnode, isDestroy);
      } catch (e) {
        handleError(e, vnode.context, ("directive " + (dir.name) + " " + hook + " hook"));
      }
    }
  }

  var baseModules = [
    ref,
    directives
  ];

  /*  */

  function updateAttrs (oldVnode, vnode) {
    var opts = vnode.componentOptions;
    if (isDef(opts) && opts.Ctor.options.inheritAttrs === false) {
      return
    }
    if (isUndef(oldVnode.data.attrs) && isUndef(vnode.data.attrs)) {
      return
    }
    var key, cur, old;
    var elm = vnode.elm;
    var oldAttrs = oldVnode.data.attrs || {};
    var attrs = vnode.data.attrs || {};
    // clone observed objects, as the user probably wants to mutate it
    if (isDef(attrs.__ob__)) {
      attrs = vnode.data.attrs = extend({}, attrs);
    }

    for (key in attrs) {
      cur = attrs[key];
      old = oldAttrs[key];
      if (old !== cur) {
        setAttr(elm, key, cur);
      }
    }
    // #4391: in IE9, setting type can reset value for input[type=radio]
    // #6666: IE/Edge forces progress value down to 1 before setting a max
    /* istanbul ignore if */
    if ((isIE || isEdge) && attrs.value !== oldAttrs.value) {
      setAttr(elm, 'value', attrs.value);
    }
    for (key in oldAttrs) {
      if (isUndef(attrs[key])) {
        if (isXlink(key)) {
          elm.removeAttributeNS(xlinkNS, getXlinkProp(key));
        } else if (!isEnumeratedAttr(key)) {
          elm.removeAttribute(key);
        }
      }
    }
  }

  function setAttr (el, key, value) {
    if (el.tagName.indexOf('-') > -1) {
      baseSetAttr(el, key, value);
    } else if (isBooleanAttr(key)) {
      // set attribute for blank value
      // e.g. <option disabled>Select one</option>
      if (isFalsyAttrValue(value)) {
        el.removeAttribute(key);
      } else {
        // technically allowfullscreen is a boolean attribute for <iframe>,
        // but Flash expects a value of "true" when used on <embed> tag
        value = key === 'allowfullscreen' && el.tagName === 'EMBED'
          ? 'true'
          : key;
        el.setAttribute(key, value);
      }
    } else if (isEnumeratedAttr(key)) {
      el.setAttribute(key, convertEnumeratedValue(key, value));
    } else if (isXlink(key)) {
      if (isFalsyAttrValue(value)) {
        el.removeAttributeNS(xlinkNS, getXlinkProp(key));
      } else {
        el.setAttributeNS(xlinkNS, key, value);
      }
    } else {
      baseSetAttr(el, key, value);
    }
  }

  function baseSetAttr (el, key, value) {
    if (isFalsyAttrValue(value)) {
      el.removeAttribute(key);
    } else {
      // #7138: IE10 & 11 fires input event when setting placeholder on
      // <textarea>... block the first input event and remove the blocker
      // immediately.
      /* istanbul ignore if */
      if (
        isIE && !isIE9 &&
        el.tagName === 'TEXTAREA' &&
        key === 'placeholder' && value !== '' && !el.__ieph
      ) {
        var blocker = function (e) {
          e.stopImmediatePropagation();
          el.removeEventListener('input', blocker);
        };
        el.addEventListener('input', blocker);
        // $flow-disable-line
        el.__ieph = true; /* IE placeholder patched */
      }
      el.setAttribute(key, value);
    }
  }

  var attrs = {
    create: updateAttrs,
    update: updateAttrs
  };

  /*  */

  function updateClass (oldVnode, vnode) {
    var el = vnode.elm;
    var data = vnode.data;
    var oldData = oldVnode.data;
    if (
      isUndef(data.staticClass) &&
      isUndef(data.class) && (
        isUndef(oldData) || (
          isUndef(oldData.staticClass) &&
          isUndef(oldData.class)
        )
      )
    ) {
      return
    }

    var cls = genClassForVnode(vnode);

    // handle transition classes
    var transitionClass = el._transitionClasses;
    if (isDef(transitionClass)) {
      cls = concat(cls, stringifyClass(transitionClass));
    }

    // set the class
    if (cls !== el._prevClass) {
      el.setAttribute('class', cls);
      el._prevClass = cls;
    }
  }

  var klass = {
    create: updateClass,
    update: updateClass
  };

  /*  */

  /*  */

  /*  */

  /*  */

  // in some cases, the event used has to be determined at runtime
  // so we used some reserved tokens during compile.
  var RANGE_TOKEN = '__r';
  var CHECKBOX_RADIO_TOKEN = '__c';

  /*  */

  // normalize v-model event tokens that can only be determined at runtime.
  // it's important to place the event as the first in the array because
  // the whole point is ensuring the v-model callback gets called before
  // user-attached handlers.
  function normalizeEvents (on) {
    /* istanbul ignore if */
    if (isDef(on[RANGE_TOKEN])) {
      // IE input[type=range] only supports `change` event
      var event = isIE ? 'change' : 'input';
      on[event] = [].concat(on[RANGE_TOKEN], on[event] || []);
      delete on[RANGE_TOKEN];
    }
    // This was originally intended to fix #4521 but no longer necessary
    // after 2.5. Keeping it for backwards compat with generated code from < 2.4
    /* istanbul ignore if */
    if (isDef(on[CHECKBOX_RADIO_TOKEN])) {
      on.change = [].concat(on[CHECKBOX_RADIO_TOKEN], on.change || []);
      delete on[CHECKBOX_RADIO_TOKEN];
    }
  }

  var target$1;

  function createOnceHandler$1 (event, handler, capture) {
    var _target = target$1; // save current target element in closure
    return function onceHandler () {
      var res = handler.apply(null, arguments);
      if (res !== null) {
        remove$2(event, onceHandler, capture, _target);
      }
    }
  }

  // #9446: Firefox <= 53 (in particular, ESR 52) has incorrect Event.timeStamp
  // implementation and does not fire microtasks in between event propagation, so
  // safe to exclude.
  var useMicrotaskFix = isUsingMicroTask && !(isFF && Number(isFF[1]) <= 53);

  function add$1 (
    name,
    handler,
    capture,
    passive
  ) {
    // async edge case #6566: inner click event triggers patch, event handler
    // attached to outer element during patch, and triggered again. This
    // happens because browsers fire microtask ticks between event propagation.
    // the solution is simple: we save the timestamp when a handler is attached,
    // and the handler would only fire if the event passed to it was fired
    // AFTER it was attached.
    if (useMicrotaskFix) {
      var attachedTimestamp = currentFlushTimestamp;
      var original = handler;
      handler = original._wrapper = function (e) {
        if (
          // no bubbling, should always fire.
          // this is just a safety net in case event.timeStamp is unreliable in
          // certain weird environments...
          e.target === e.currentTarget ||
          // event is fired after handler attachment
          e.timeStamp >= attachedTimestamp ||
          // bail for environments that have buggy event.timeStamp implementations
          // #9462 iOS 9 bug: event.timeStamp is 0 after history.pushState
          // #9681 QtWebEngine event.timeStamp is negative value
          e.timeStamp <= 0 ||
          // #9448 bail if event is fired in another document in a multi-page
          // electron/nw.js app, since event.timeStamp will be using a different
          // starting reference
          e.target.ownerDocument !== document
        ) {
          return original.apply(this, arguments)
        }
      };
    }
    target$1.addEventListener(
      name,
      handler,
      supportsPassive
        ? { capture: capture, passive: passive }
        : capture
    );
  }

  function remove$2 (
    name,
    handler,
    capture,
    _target
  ) {
    (_target || target$1).removeEventListener(
      name,
      handler._wrapper || handler,
      capture
    );
  }

  function updateDOMListeners (oldVnode, vnode) {
    if (isUndef(oldVnode.data.on) && isUndef(vnode.data.on)) {
      return
    }
    var on = vnode.data.on || {};
    var oldOn = oldVnode.data.on || {};
    target$1 = vnode.elm;
    normalizeEvents(on);
    updateListeners(on, oldOn, add$1, remove$2, createOnceHandler$1, vnode.context);
    target$1 = undefined;
  }

  var events = {
    create: updateDOMListeners,
    update: updateDOMListeners
  };

  /*  */

  var svgContainer;

  function updateDOMProps (oldVnode, vnode) {
    if (isUndef(oldVnode.data.domProps) && isUndef(vnode.data.domProps)) {
      return
    }
    var key, cur;
    var elm = vnode.elm;
    var oldProps = oldVnode.data.domProps || {};
    var props = vnode.data.domProps || {};
    // clone observed objects, as the user probably wants to mutate it
    if (isDef(props.__ob__)) {
      props = vnode.data.domProps = extend({}, props);
    }

    for (key in oldProps) {
      if (!(key in props)) {
        elm[key] = '';
      }
    }

    for (key in props) {
      cur = props[key];
      // ignore children if the node has textContent or innerHTML,
      // as these will throw away existing DOM nodes and cause removal errors
      // on subsequent patches (#3360)
      if (key === 'textContent' || key === 'innerHTML') {
        if (vnode.children) { vnode.children.length = 0; }
        if (cur === oldProps[key]) { continue }
        // #6601 work around Chrome version <= 55 bug where single textNode
        // replaced by innerHTML/textContent retains its parentNode property
        if (elm.childNodes.length === 1) {
          elm.removeChild(elm.childNodes[0]);
        }
      }

      if (key === 'value' && elm.tagName !== 'PROGRESS') {
        // store value as _value as well since
        // non-string values will be stringified
        elm._value = cur;
        // avoid resetting cursor position when value is the same
        var strCur = isUndef(cur) ? '' : String(cur);
        if (shouldUpdateValue(elm, strCur)) {
          elm.value = strCur;
        }
      } else if (key === 'innerHTML' && isSVG(elm.tagName) && isUndef(elm.innerHTML)) {
        // IE doesn't support innerHTML for SVG elements
        svgContainer = svgContainer || document.createElement('div');
        svgContainer.innerHTML = "<svg>" + cur + "</svg>";
        var svg = svgContainer.firstChild;
        while (elm.firstChild) {
          elm.removeChild(elm.firstChild);
        }
        while (svg.firstChild) {
          elm.appendChild(svg.firstChild);
        }
      } else if (
        // skip the update if old and new VDOM state is the same.
        // `value` is handled separately because the DOM value may be temporarily
        // out of sync with VDOM state due to focus, composition and modifiers.
        // This  #4521 by skipping the unnecesarry `checked` update.
        cur !== oldProps[key]
      ) {
        // some property updates can throw
        // e.g. `value` on <progress> w/ non-finite value
        try {
          elm[key] = cur;
        } catch (e) {}
      }
    }
  }

  // check platforms/web/util/attrs.js acceptValue


  function shouldUpdateValue (elm, checkVal) {
    return (!elm.composing && (
      elm.tagName === 'OPTION' ||
      isNotInFocusAndDirty(elm, checkVal) ||
      isDirtyWithModifiers(elm, checkVal)
    ))
  }

  function isNotInFocusAndDirty (elm, checkVal) {
    // return true when textbox (.number and .trim) loses focus and its value is
    // not equal to the updated value
    var notInFocus = true;
    // #6157
    // work around IE bug when accessing document.activeElement in an iframe
    try { notInFocus = document.activeElement !== elm; } catch (e) {}
    return notInFocus && elm.value !== checkVal
  }

  function isDirtyWithModifiers (elm, newVal) {
    var value = elm.value;
    var modifiers = elm._vModifiers; // injected by v-model runtime
    if (isDef(modifiers)) {
      if (modifiers.number) {
        return toNumber(value) !== toNumber(newVal)
      }
      if (modifiers.trim) {
        return value.trim() !== newVal.trim()
      }
    }
    return value !== newVal
  }

  var domProps = {
    create: updateDOMProps,
    update: updateDOMProps
  };

  /*  */

  var parseStyleText = cached(function (cssText) {
    var res = {};
    var listDelimiter = /;(?![^(]*\))/g;
    var propertyDelimiter = /:(.+)/;
    cssText.split(listDelimiter).forEach(function (item) {
      if (item) {
        var tmp = item.split(propertyDelimiter);
        tmp.length > 1 && (res[tmp[0].trim()] = tmp[1].trim());
      }
    });
    return res
  });

  // merge static and dynamic style data on the same vnode
  function normalizeStyleData (data) {
    var style = normalizeStyleBinding(data.style);
    // static style is pre-processed into an object during compilation
    // and is always a fresh object, so it's safe to merge into it
    return data.staticStyle
      ? extend(data.staticStyle, style)
      : style
  }

  // normalize possible array / string values into Object
  function normalizeStyleBinding (bindingStyle) {
    if (Array.isArray(bindingStyle)) {
      return toObject(bindingStyle)
    }
    if (typeof bindingStyle === 'string') {
      return parseStyleText(bindingStyle)
    }
    return bindingStyle
  }

  /**
   * parent component style should be after child's
   * so that parent component's style could override it
   */
  function getStyle (vnode, checkChild) {
    var res = {};
    var styleData;

    if (checkChild) {
      var childNode = vnode;
      while (childNode.componentInstance) {
        childNode = childNode.componentInstance._vnode;
        if (
          childNode && childNode.data &&
          (styleData = normalizeStyleData(childNode.data))
        ) {
          extend(res, styleData);
        }
      }
    }

    if ((styleData = normalizeStyleData(vnode.data))) {
      extend(res, styleData);
    }

    var parentNode = vnode;
    while ((parentNode = parentNode.parent)) {
      if (parentNode.data && (styleData = normalizeStyleData(parentNode.data))) {
        extend(res, styleData);
      }
    }
    return res
  }

  /*  */

  var cssVarRE = /^--/;
  var importantRE = /\s*!important$/;
  var setProp = function (el, name, val) {
    /* istanbul ignore if */
    if (cssVarRE.test(name)) {
      el.style.setProperty(name, val);
    } else if (importantRE.test(val)) {
      el.style.setProperty(hyphenate(name), val.replace(importantRE, ''), 'important');
    } else {
      var normalizedName = normalize(name);
      if (Array.isArray(val)) {
        // Support values array created by autoprefixer, e.g.
        // {display: ["-webkit-box", "-ms-flexbox", "flex"]}
        // Set them one by one, and the browser will only set those it can recognize
        for (var i = 0, len = val.length; i < len; i++) {
          el.style[normalizedName] = val[i];
        }
      } else {
        el.style[normalizedName] = val;
      }
    }
  };

  var vendorNames = ['Webkit', 'Moz', 'ms'];

  var emptyStyle;
  var normalize = cached(function (prop) {
    emptyStyle = emptyStyle || document.createElement('div').style;
    prop = camelize(prop);
    if (prop !== 'filter' && (prop in emptyStyle)) {
      return prop
    }
    var capName = prop.charAt(0).toUpperCase() + prop.slice(1);
    for (var i = 0; i < vendorNames.length; i++) {
      var name = vendorNames[i] + capName;
      if (name in emptyStyle) {
        return name
      }
    }
  });

  function updateStyle (oldVnode, vnode) {
    var data = vnode.data;
    var oldData = oldVnode.data;

    if (isUndef(data.staticStyle) && isUndef(data.style) &&
      isUndef(oldData.staticStyle) && isUndef(oldData.style)
    ) {
      return
    }

    var cur, name;
    var el = vnode.elm;
    var oldStaticStyle = oldData.staticStyle;
    var oldStyleBinding = oldData.normalizedStyle || oldData.style || {};

    // if static style exists, stylebinding already merged into it when doing normalizeStyleData
    var oldStyle = oldStaticStyle || oldStyleBinding;

    var style = normalizeStyleBinding(vnode.data.style) || {};

    // store normalized style under a different key for next diff
    // make sure to clone it if it's reactive, since the user likely wants
    // to mutate it.
    vnode.data.normalizedStyle = isDef(style.__ob__)
      ? extend({}, style)
      : style;

    var newStyle = getStyle(vnode, true);

    for (name in oldStyle) {
      if (isUndef(newStyle[name])) {
        setProp(el, name, '');
      }
    }
    for (name in newStyle) {
      cur = newStyle[name];
      if (cur !== oldStyle[name]) {
        // ie9 setting to null has no effect, must use empty string
        setProp(el, name, cur == null ? '' : cur);
      }
    }
  }

  var style = {
    create: updateStyle,
    update: updateStyle
  };

  /*  */

  var whitespaceRE = /\s+/;

  /**
   * Add class with compatibility for SVG since classList is not supported on
   * SVG elements in IE
   */
  function addClass (el, cls) {
    /* istanbul ignore if */
    if (!cls || !(cls = cls.trim())) {
      return
    }

    /* istanbul ignore else */
    if (el.classList) {
      if (cls.indexOf(' ') > -1) {
        cls.split(whitespaceRE).forEach(function (c) { return el.classList.add(c); });
      } else {
        el.classList.add(cls);
      }
    } else {
      var cur = " " + (el.getAttribute('class') || '') + " ";
      if (cur.indexOf(' ' + cls + ' ') < 0) {
        el.setAttribute('class', (cur + cls).trim());
      }
    }
  }

  /**
   * Remove class with compatibility for SVG since classList is not supported on
   * SVG elements in IE
   */
  function removeClass (el, cls) {
    /* istanbul ignore if */
    if (!cls || !(cls = cls.trim())) {
      return
    }

    /* istanbul ignore else */
    if (el.classList) {
      if (cls.indexOf(' ') > -1) {
        cls.split(whitespaceRE).forEach(function (c) { return el.classList.remove(c); });
      } else {
        el.classList.remove(cls);
      }
      if (!el.classList.length) {
        el.removeAttribute('class');
      }
    } else {
      var cur = " " + (el.getAttribute('class') || '') + " ";
      var tar = ' ' + cls + ' ';
      while (cur.indexOf(tar) >= 0) {
        cur = cur.replace(tar, ' ');
      }
      cur = cur.trim();
      if (cur) {
        el.setAttribute('class', cur);
      } else {
        el.removeAttribute('class');
      }
    }
  }

  /*  */

  function resolveTransition (def$$1) {
    if (!def$$1) {
      return
    }
    /* istanbul ignore else */
    if (typeof def$$1 === 'object') {
      var res = {};
      if (def$$1.css !== false) {
        extend(res, autoCssTransition(def$$1.name || 'v'));
      }
      extend(res, def$$1);
      return res
    } else if (typeof def$$1 === 'string') {
      return autoCssTransition(def$$1)
    }
  }

  var autoCssTransition = cached(function (name) {
    return {
      enterClass: (name + "-enter"),
      enterToClass: (name + "-enter-to"),
      enterActiveClass: (name + "-enter-active"),
      leaveClass: (name + "-leave"),
      leaveToClass: (name + "-leave-to"),
      leaveActiveClass: (name + "-leave-active")
    }
  });

  var hasTransition = inBrowser && !isIE9;
  var TRANSITION = 'transition';
  var ANIMATION = 'animation';

  // Transition property/event sniffing
  var transitionProp = 'transition';
  var transitionEndEvent = 'transitionend';
  var animationProp = 'animation';
  var animationEndEvent = 'animationend';
  if (hasTransition) {
    /* istanbul ignore if */
    if (window.ontransitionend === undefined &&
      window.onwebkittransitionend !== undefined
    ) {
      transitionProp = 'WebkitTransition';
      transitionEndEvent = 'webkitTransitionEnd';
    }
    if (window.onanimationend === undefined &&
      window.onwebkitanimationend !== undefined
    ) {
      animationProp = 'WebkitAnimation';
      animationEndEvent = 'webkitAnimationEnd';
    }
  }

  // binding to window is necessary to make hot reload work in IE in strict mode
  var raf = inBrowser
    ? window.requestAnimationFrame
      ? window.requestAnimationFrame.bind(window)
      : setTimeout
    : /* istanbul ignore next */ function (fn) { return fn(); };

  function nextFrame (fn) {
    raf(function () {
      raf(fn);
    });
  }

  function addTransitionClass (el, cls) {
    var transitionClasses = el._transitionClasses || (el._transitionClasses = []);
    if (transitionClasses.indexOf(cls) < 0) {
      transitionClasses.push(cls);
      addClass(el, cls);
    }
  }

  function removeTransitionClass (el, cls) {
    if (el._transitionClasses) {
      remove(el._transitionClasses, cls);
    }
    removeClass(el, cls);
  }

  function whenTransitionEnds (
    el,
    expectedType,
    cb
  ) {
    var ref = getTransitionInfo(el, expectedType);
    var type = ref.type;
    var timeout = ref.timeout;
    var propCount = ref.propCount;
    if (!type) { return cb() }
    var event = type === TRANSITION ? transitionEndEvent : animationEndEvent;
    var ended = 0;
    var end = function () {
      el.removeEventListener(event, onEnd);
      cb();
    };
    var onEnd = function (e) {
      if (e.target === el) {
        if (++ended >= propCount) {
          end();
        }
      }
    };
    setTimeout(function () {
      if (ended < propCount) {
        end();
      }
    }, timeout + 1);
    el.addEventListener(event, onEnd);
  }

  var transformRE = /\b(transform|all)(,|$)/;

  function getTransitionInfo (el, expectedType) {
    var styles = window.getComputedStyle(el);
    // JSDOM may return undefined for transition properties
    var transitionDelays = (styles[transitionProp + 'Delay'] || '').split(', ');
    var transitionDurations = (styles[transitionProp + 'Duration'] || '').split(', ');
    var transitionTimeout = getTimeout(transitionDelays, transitionDurations);
    var animationDelays = (styles[animationProp + 'Delay'] || '').split(', ');
    var animationDurations = (styles[animationProp + 'Duration'] || '').split(', ');
    var animationTimeout = getTimeout(animationDelays, animationDurations);

    var type;
    var timeout = 0;
    var propCount = 0;
    /* istanbul ignore if */
    if (expectedType === TRANSITION) {
      if (transitionTimeout > 0) {
        type = TRANSITION;
        timeout = transitionTimeout;
        propCount = transitionDurations.length;
      }
    } else if (expectedType === ANIMATION) {
      if (animationTimeout > 0) {
        type = ANIMATION;
        timeout = animationTimeout;
        propCount = animationDurations.length;
      }
    } else {
      timeout = Math.max(transitionTimeout, animationTimeout);
      type = timeout > 0
        ? transitionTimeout > animationTimeout
          ? TRANSITION
          : ANIMATION
        : null;
      propCount = type
        ? type === TRANSITION
          ? transitionDurations.length
          : animationDurations.length
        : 0;
    }
    var hasTransform =
      type === TRANSITION &&
      transformRE.test(styles[transitionProp + 'Property']);
    return {
      type: type,
      timeout: timeout,
      propCount: propCount,
      hasTransform: hasTransform
    }
  }

  function getTimeout (delays, durations) {
    /* istanbul ignore next */
    while (delays.length < durations.length) {
      delays = delays.concat(delays);
    }

    return Math.max.apply(null, durations.map(function (d, i) {
      return toMs(d) + toMs(delays[i])
    }))
  }

  // Old versions of Chromium (below 61.0.3163.100) formats floating pointer numbers
  // in a locale-dependent way, using a comma instead of a dot.
  // If comma is not replaced with a dot, the input will be rounded down (i.e. acting
  // as a floor function) causing unexpected behaviors
  function toMs (s) {
    return Number(s.slice(0, -1).replace(',', '.')) * 1000
  }

  /*  */

  function enter (vnode, toggleDisplay) {
    var el = vnode.elm;

    // call leave callback now
    if (isDef(el._leaveCb)) {
      el._leaveCb.cancelled = true;
      el._leaveCb();
    }

    var data = resolveTransition(vnode.data.transition);
    if (isUndef(data)) {
      return
    }

    /* istanbul ignore if */
    if (isDef(el._enterCb) || el.nodeType !== 1) {
      return
    }

    var css = data.css;
    var type = data.type;
    var enterClass = data.enterClass;
    var enterToClass = data.enterToClass;
    var enterActiveClass = data.enterActiveClass;
    var appearClass = data.appearClass;
    var appearToClass = data.appearToClass;
    var appearActiveClass = data.appearActiveClass;
    var beforeEnter = data.beforeEnter;
    var enter = data.enter;
    var afterEnter = data.afterEnter;
    var enterCancelled = data.enterCancelled;
    var beforeAppear = data.beforeAppear;
    var appear = data.appear;
    var afterAppear = data.afterAppear;
    var appearCancelled = data.appearCancelled;
    var duration = data.duration;

    // activeInstance will always be the <transition> component managing this
    // transition. One edge case to check is when the <transition> is placed
    // as the root node of a child component. In that case we need to check
    // <transition>'s parent for appear check.
    var context = activeInstance;
    var transitionNode = activeInstance.$vnode;
    while (transitionNode && transitionNode.parent) {
      context = transitionNode.context;
      transitionNode = transitionNode.parent;
    }

    var isAppear = !context._isMounted || !vnode.isRootInsert;

    if (isAppear && !appear && appear !== '') {
      return
    }

    var startClass = isAppear && appearClass
      ? appearClass
      : enterClass;
    var activeClass = isAppear && appearActiveClass
      ? appearActiveClass
      : enterActiveClass;
    var toClass = isAppear && appearToClass
      ? appearToClass
      : enterToClass;

    var beforeEnterHook = isAppear
      ? (beforeAppear || beforeEnter)
      : beforeEnter;
    var enterHook = isAppear
      ? (typeof appear === 'function' ? appear : enter)
      : enter;
    var afterEnterHook = isAppear
      ? (afterAppear || afterEnter)
      : afterEnter;
    var enterCancelledHook = isAppear
      ? (appearCancelled || enterCancelled)
      : enterCancelled;

    var explicitEnterDuration = toNumber(
      isObject(duration)
        ? duration.enter
        : duration
    );

    if (explicitEnterDuration != null) {
      checkDuration(explicitEnterDuration, 'enter', vnode);
    }

    var expectsCSS = css !== false && !isIE9;
    var userWantsControl = getHookArgumentsLength(enterHook);

    var cb = el._enterCb = once(function () {
      if (expectsCSS) {
        removeTransitionClass(el, toClass);
        removeTransitionClass(el, activeClass);
      }
      if (cb.cancelled) {
        if (expectsCSS) {
          removeTransitionClass(el, startClass);
        }
        enterCancelledHook && enterCancelledHook(el);
      } else {
        afterEnterHook && afterEnterHook(el);
      }
      el._enterCb = null;
    });

    if (!vnode.data.show) {
      // remove pending leave element on enter by injecting an insert hook
      mergeVNodeHook(vnode, 'insert', function () {
        var parent = el.parentNode;
        var pendingNode = parent && parent._pending && parent._pending[vnode.key];
        if (pendingNode &&
          pendingNode.tag === vnode.tag &&
          pendingNode.elm._leaveCb
        ) {
          pendingNode.elm._leaveCb();
        }
        enterHook && enterHook(el, cb);
      });
    }

    // start enter transition
    beforeEnterHook && beforeEnterHook(el);
    if (expectsCSS) {
      addTransitionClass(el, startClass);
      addTransitionClass(el, activeClass);
      nextFrame(function () {
        removeTransitionClass(el, startClass);
        if (!cb.cancelled) {
          addTransitionClass(el, toClass);
          if (!userWantsControl) {
            if (isValidDuration(explicitEnterDuration)) {
              setTimeout(cb, explicitEnterDuration);
            } else {
              whenTransitionEnds(el, type, cb);
            }
          }
        }
      });
    }

    if (vnode.data.show) {
      toggleDisplay && toggleDisplay();
      enterHook && enterHook(el, cb);
    }

    if (!expectsCSS && !userWantsControl) {
      cb();
    }
  }

  function leave (vnode, rm) {
    var el = vnode.elm;

    // call enter callback now
    if (isDef(el._enterCb)) {
      el._enterCb.cancelled = true;
      el._enterCb();
    }

    var data = resolveTransition(vnode.data.transition);
    if (isUndef(data) || el.nodeType !== 1) {
      return rm()
    }

    /* istanbul ignore if */
    if (isDef(el._leaveCb)) {
      return
    }

    var css = data.css;
    var type = data.type;
    var leaveClass = data.leaveClass;
    var leaveToClass = data.leaveToClass;
    var leaveActiveClass = data.leaveActiveClass;
    var beforeLeave = data.beforeLeave;
    var leave = data.leave;
    var afterLeave = data.afterLeave;
    var leaveCancelled = data.leaveCancelled;
    var delayLeave = data.delayLeave;
    var duration = data.duration;

    var expectsCSS = css !== false && !isIE9;
    var userWantsControl = getHookArgumentsLength(leave);

    var explicitLeaveDuration = toNumber(
      isObject(duration)
        ? duration.leave
        : duration
    );

    if (isDef(explicitLeaveDuration)) {
      checkDuration(explicitLeaveDuration, 'leave', vnode);
    }

    var cb = el._leaveCb = once(function () {
      if (el.parentNode && el.parentNode._pending) {
        el.parentNode._pending[vnode.key] = null;
      }
      if (expectsCSS) {
        removeTransitionClass(el, leaveToClass);
        removeTransitionClass(el, leaveActiveClass);
      }
      if (cb.cancelled) {
        if (expectsCSS) {
          removeTransitionClass(el, leaveClass);
        }
        leaveCancelled && leaveCancelled(el);
      } else {
        rm();
        afterLeave && afterLeave(el);
      }
      el._leaveCb = null;
    });

    if (delayLeave) {
      delayLeave(performLeave);
    } else {
      performLeave();
    }

    function performLeave () {
      // the delayed leave may have already been cancelled
      if (cb.cancelled) {
        return
      }
      // record leaving element
      if (!vnode.data.show && el.parentNode) {
        (el.parentNode._pending || (el.parentNode._pending = {}))[(vnode.key)] = vnode;
      }
      beforeLeave && beforeLeave(el);
      if (expectsCSS) {
        addTransitionClass(el, leaveClass);
        addTransitionClass(el, leaveActiveClass);
        nextFrame(function () {
          removeTransitionClass(el, leaveClass);
          if (!cb.cancelled) {
            addTransitionClass(el, leaveToClass);
            if (!userWantsControl) {
              if (isValidDuration(explicitLeaveDuration)) {
                setTimeout(cb, explicitLeaveDuration);
              } else {
                whenTransitionEnds(el, type, cb);
              }
            }
          }
        });
      }
      leave && leave(el, cb);
      if (!expectsCSS && !userWantsControl) {
        cb();
      }
    }
  }

  // only used in dev mode
  function checkDuration (val, name, vnode) {
    if (typeof val !== 'number') {
      warn(
        "<transition> explicit " + name + " duration is not a valid number - " +
        "got " + (JSON.stringify(val)) + ".",
        vnode.context
      );
    } else if (isNaN(val)) {
      warn(
        "<transition> explicit " + name + " duration is NaN - " +
        'the duration expression might be incorrect.',
        vnode.context
      );
    }
  }

  function isValidDuration (val) {
    return typeof val === 'number' && !isNaN(val)
  }

  /**
   * Normalize a transition hook's argument length. The hook may be:
   * - a merged hook (invoker) with the original in .fns
   * - a wrapped component method (check ._length)
   * - a plain function (.length)
   */
  function getHookArgumentsLength (fn) {
    if (isUndef(fn)) {
      return false
    }
    var invokerFns = fn.fns;
    if (isDef(invokerFns)) {
      // invoker
      return getHookArgumentsLength(
        Array.isArray(invokerFns)
          ? invokerFns[0]
          : invokerFns
      )
    } else {
      return (fn._length || fn.length) > 1
    }
  }

  function _enter (_, vnode) {
    if (vnode.data.show !== true) {
      enter(vnode);
    }
  }

  var transition = inBrowser ? {
    create: _enter,
    activate: _enter,
    remove: function remove$$1 (vnode, rm) {
      /* istanbul ignore else */
      if (vnode.data.show !== true) {
        leave(vnode, rm);
      } else {
        rm();
      }
    }
  } : {};

  var platformModules = [
    attrs,
    klass,
    events,
    domProps,
    style,
    transition
  ];

  /*  */

  // the directive module should be applied last, after all
  // built-in modules have been applied.
  var modules = platformModules.concat(baseModules);

  var patch = createPatchFunction({ nodeOps: nodeOps, modules: modules });

  /**
   * Not type checking this file because flow doesn't like attaching
   * properties to Elements.
   */

  /* istanbul ignore if */
  if (isIE9) {
    // http://www.matts411.com/post/internet-explorer-9-oninput/
    document.addEventListener('selectionchange', function () {
      var el = document.activeElement;
      if (el && el.vmodel) {
        trigger(el, 'input');
      }
    });
  }

  var directive = {
    inserted: function inserted (el, binding, vnode, oldVnode) {
      if (vnode.tag === 'select') {
        // #6903
        if (oldVnode.elm && !oldVnode.elm._vOptions) {
          mergeVNodeHook(vnode, 'postpatch', function () {
            directive.componentUpdated(el, binding, vnode);
          });
        } else {
          setSelected(el, binding, vnode.context);
        }
        el._vOptions = [].map.call(el.options, getValue);
      } else if (vnode.tag === 'textarea' || isTextInputType(el.type)) {
        el._vModifiers = binding.modifiers;
        if (!binding.modifiers.lazy) {
          el.addEventListener('compositionstart', onCompositionStart);
          el.addEventListener('compositionend', onCompositionEnd);
          // Safari < 10.2 & UIWebView doesn't fire compositionend when
          // switching focus before confirming composition choice
          // this also fixes the issue where some browsers e.g. iOS Chrome
          // fires "change" instead of "input" on autocomplete.
          el.addEventListener('change', onCompositionEnd);
          /* istanbul ignore if */
          if (isIE9) {
            el.vmodel = true;
          }
        }
      }
    },

    componentUpdated: function componentUpdated (el, binding, vnode) {
      if (vnode.tag === 'select') {
        setSelected(el, binding, vnode.context);
        // in case the options rendered by v-for have changed,
        // it's possible that the value is out-of-sync with the rendered options.
        // detect such cases and filter out values that no longer has a matching
        // option in the DOM.
        var prevOptions = el._vOptions;
        var curOptions = el._vOptions = [].map.call(el.options, getValue);
        if (curOptions.some(function (o, i) { return !looseEqual(o, prevOptions[i]); })) {
          // trigger change event if
          // no matching option found for at least one value
          var needReset = el.multiple
            ? binding.value.some(function (v) { return hasNoMatchingOption(v, curOptions); })
            : binding.value !== binding.oldValue && hasNoMatchingOption(binding.value, curOptions);
          if (needReset) {
            trigger(el, 'change');
          }
        }
      }
    }
  };

  function setSelected (el, binding, vm) {
    actuallySetSelected(el, binding, vm);
    /* istanbul ignore if */
    if (isIE || isEdge) {
      setTimeout(function () {
        actuallySetSelected(el, binding, vm);
      }, 0);
    }
  }

  function actuallySetSelected (el, binding, vm) {
    var value = binding.value;
    var isMultiple = el.multiple;
    if (isMultiple && !Array.isArray(value)) {
      warn(
        "<select multiple v-model=\"" + (binding.expression) + "\"> " +
        "expects an Array value for its binding, but got " + (Object.prototype.toString.call(value).slice(8, -1)),
        vm
      );
      return
    }
    var selected, option;
    for (var i = 0, l = el.options.length; i < l; i++) {
      option = el.options[i];
      if (isMultiple) {
        selected = looseIndexOf(value, getValue(option)) > -1;
        if (option.selected !== selected) {
          option.selected = selected;
        }
      } else {
        if (looseEqual(getValue(option), value)) {
          if (el.selectedIndex !== i) {
            el.selectedIndex = i;
          }
          return
        }
      }
    }
    if (!isMultiple) {
      el.selectedIndex = -1;
    }
  }

  function hasNoMatchingOption (value, options) {
    return options.every(function (o) { return !looseEqual(o, value); })
  }

  function getValue (option) {
    return '_value' in option
      ? option._value
      : option.value
  }

  function onCompositionStart (e) {
    e.target.composing = true;
  }

  function onCompositionEnd (e) {
    // prevent triggering an input event for no reason
    if (!e.target.composing) { return }
    e.target.composing = false;
    trigger(e.target, 'input');
  }

  function trigger (el, type) {
    var e = document.createEvent('HTMLEvents');
    e.initEvent(type, true, true);
    el.dispatchEvent(e);
  }

  /*  */

  // recursively search for possible transition defined inside the component root
  function locateNode (vnode) {
    return vnode.componentInstance && (!vnode.data || !vnode.data.transition)
      ? locateNode(vnode.componentInstance._vnode)
      : vnode
  }

  var show = {
    bind: function bind (el, ref, vnode) {
      var value = ref.value;

      vnode = locateNode(vnode);
      var transition$$1 = vnode.data && vnode.data.transition;
      var originalDisplay = el.__vOriginalDisplay =
        el.style.display === 'none' ? '' : el.style.display;
      if (value && transition$$1) {
        vnode.data.show = true;
        enter(vnode, function () {
          el.style.display = originalDisplay;
        });
      } else {
        el.style.display = value ? originalDisplay : 'none';
      }
    },

    update: function update (el, ref, vnode) {
      var value = ref.value;
      var oldValue = ref.oldValue;

      /* istanbul ignore if */
      if (!value === !oldValue) { return }
      vnode = locateNode(vnode);
      var transition$$1 = vnode.data && vnode.data.transition;
      if (transition$$1) {
        vnode.data.show = true;
        if (value) {
          enter(vnode, function () {
            el.style.display = el.__vOriginalDisplay;
          });
        } else {
          leave(vnode, function () {
            el.style.display = 'none';
          });
        }
      } else {
        el.style.display = value ? el.__vOriginalDisplay : 'none';
      }
    },

    unbind: function unbind (
      el,
      binding,
      vnode,
      oldVnode,
      isDestroy
    ) {
      if (!isDestroy) {
        el.style.display = el.__vOriginalDisplay;
      }
    }
  };

  var platformDirectives = {
    model: directive,
    show: show
  };

  /*  */

  var transitionProps = {
    name: String,
    appear: Boolean,
    css: Boolean,
    mode: String,
    type: String,
    enterClass: String,
    leaveClass: String,
    enterToClass: String,
    leaveToClass: String,
    enterActiveClass: String,
    leaveActiveClass: String,
    appearClass: String,
    appearActiveClass: String,
    appearToClass: String,
    duration: [Number, String, Object]
  };

  // in case the child is also an abstract component, e.g. <keep-alive>
  // we want to recursively retrieve the real component to be rendered
  function getRealChild (vnode) {
    var compOptions = vnode && vnode.componentOptions;
    if (compOptions && compOptions.Ctor.options.abstract) {
      return getRealChild(getFirstComponentChild(compOptions.children))
    } else {
      return vnode
    }
  }

  function extractTransitionData (comp) {
    var data = {};
    var options = comp.$options;
    // props
    for (var key in options.propsData) {
      data[key] = comp[key];
    }
    // events.
    // extract listeners and pass them directly to the transition methods
    var listeners = options._parentListeners;
    for (var key$1 in listeners) {
      data[camelize(key$1)] = listeners[key$1];
    }
    return data
  }

  function placeholder (h, rawChild) {
    if (/\d-keep-alive$/.test(rawChild.tag)) {
      return h('keep-alive', {
        props: rawChild.componentOptions.propsData
      })
    }
  }

  function hasParentTransition (vnode) {
    while ((vnode = vnode.parent)) {
      if (vnode.data.transition) {
        return true
      }
    }
  }

  function isSameChild (child, oldChild) {
    return oldChild.key === child.key && oldChild.tag === child.tag
  }

  var isNotTextNode = function (c) { return c.tag || isAsyncPlaceholder(c); };

  var isVShowDirective = function (d) { return d.name === 'show'; };

  var Transition = {
    name: 'transition',
    props: transitionProps,
    abstract: true,

    render: function render (h) {
      var this$1 = this;

      var children = this.$slots.default;
      if (!children) {
        return
      }

      // filter out text nodes (possible whitespaces)
      children = children.filter(isNotTextNode);
      /* istanbul ignore if */
      if (!children.length) {
        return
      }

      // warn multiple elements
      if (children.length > 1) {
        warn(
          '<transition> can only be used on a single element. Use ' +
          '<transition-group> for lists.',
          this.$parent
        );
      }

      var mode = this.mode;

      // warn invalid mode
      if (mode && mode !== 'in-out' && mode !== 'out-in'
      ) {
        warn(
          'invalid <transition> mode: ' + mode,
          this.$parent
        );
      }

      var rawChild = children[0];

      // if this is a component root node and the component's
      // parent container node also has transition, skip.
      if (hasParentTransition(this.$vnode)) {
        return rawChild
      }

      // apply transition data to child
      // use getRealChild() to ignore abstract components e.g. keep-alive
      var child = getRealChild(rawChild);
      /* istanbul ignore if */
      if (!child) {
        return rawChild
      }

      if (this._leaving) {
        return placeholder(h, rawChild)
      }

      // ensure a key that is unique to the vnode type and to this transition
      // component instance. This key will be used to remove pending leaving nodes
      // during entering.
      var id = "__transition-" + (this._uid) + "-";
      child.key = child.key == null
        ? child.isComment
          ? id + 'comment'
          : id + child.tag
        : isPrimitive(child.key)
          ? (String(child.key).indexOf(id) === 0 ? child.key : id + child.key)
          : child.key;

      var data = (child.data || (child.data = {})).transition = extractTransitionData(this);
      var oldRawChild = this._vnode;
      var oldChild = getRealChild(oldRawChild);

      // mark v-show
      // so that the transition module can hand over the control to the directive
      if (child.data.directives && child.data.directives.some(isVShowDirective)) {
        child.data.show = true;
      }

      if (
        oldChild &&
        oldChild.data &&
        !isSameChild(child, oldChild) &&
        !isAsyncPlaceholder(oldChild) &&
        // #6687 component root is a comment node
        !(oldChild.componentInstance && oldChild.componentInstance._vnode.isComment)
      ) {
        // replace old child transition data with fresh one
        // important for dynamic transitions!
        var oldData = oldChild.data.transition = extend({}, data);
        // handle transition mode
        if (mode === 'out-in') {
          // return placeholder node and queue update when leave finishes
          this._leaving = true;
          mergeVNodeHook(oldData, 'afterLeave', function () {
            this$1._leaving = false;
            this$1.$forceUpdate();
          });
          return placeholder(h, rawChild)
        } else if (mode === 'in-out') {
          if (isAsyncPlaceholder(child)) {
            return oldRawChild
          }
          var delayedLeave;
          var performLeave = function () { delayedLeave(); };
          mergeVNodeHook(data, 'afterEnter', performLeave);
          mergeVNodeHook(data, 'enterCancelled', performLeave);
          mergeVNodeHook(oldData, 'delayLeave', function (leave) { delayedLeave = leave; });
        }
      }

      return rawChild
    }
  };

  /*  */

  var props = extend({
    tag: String,
    moveClass: String
  }, transitionProps);

  delete props.mode;

  var TransitionGroup = {
    props: props,

    beforeMount: function beforeMount () {
      var this$1 = this;

      var update = this._update;
      this._update = function (vnode, hydrating) {
        var restoreActiveInstance = setActiveInstance(this$1);
        // force removing pass
        this$1.__patch__(
          this$1._vnode,
          this$1.kept,
          false, // hydrating
          true // removeOnly (!important, avoids unnecessary moves)
        );
        this$1._vnode = this$1.kept;
        restoreActiveInstance();
        update.call(this$1, vnode, hydrating);
      };
    },

    render: function render (h) {
      var tag = this.tag || this.$vnode.data.tag || 'span';
      var map = Object.create(null);
      var prevChildren = this.prevChildren = this.children;
      var rawChildren = this.$slots.default || [];
      var children = this.children = [];
      var transitionData = extractTransitionData(this);

      for (var i = 0; i < rawChildren.length; i++) {
        var c = rawChildren[i];
        if (c.tag) {
          if (c.key != null && String(c.key).indexOf('__vlist') !== 0) {
            children.push(c);
            map[c.key] = c
            ;(c.data || (c.data = {})).transition = transitionData;
          } else {
            var opts = c.componentOptions;
            var name = opts ? (opts.Ctor.options.name || opts.tag || '') : c.tag;
            warn(("<transition-group> children must be keyed: <" + name + ">"));
          }
        }
      }

      if (prevChildren) {
        var kept = [];
        var removed = [];
        for (var i$1 = 0; i$1 < prevChildren.length; i$1++) {
          var c$1 = prevChildren[i$1];
          c$1.data.transition = transitionData;
          c$1.data.pos = c$1.elm.getBoundingClientRect();
          if (map[c$1.key]) {
            kept.push(c$1);
          } else {
            removed.push(c$1);
          }
        }
        this.kept = h(tag, null, kept);
        this.removed = removed;
      }

      return h(tag, null, children)
    },

    updated: function updated () {
      var children = this.prevChildren;
      var moveClass = this.moveClass || ((this.name || 'v') + '-move');
      if (!children.length || !this.hasMove(children[0].elm, moveClass)) {
        return
      }

      // we divide the work into three loops to avoid mixing DOM reads and writes
      // in each iteration - which helps prevent layout thrashing.
      children.forEach(callPendingCbs);
      children.forEach(recordPosition);
      children.forEach(applyTranslation);

      // force reflow to put everything in position
      // assign to this to avoid being removed in tree-shaking
      // $flow-disable-line
      this._reflow = document.body.offsetHeight;

      children.forEach(function (c) {
        if (c.data.moved) {
          var el = c.elm;
          var s = el.style;
          addTransitionClass(el, moveClass);
          s.transform = s.WebkitTransform = s.transitionDuration = '';
          el.addEventListener(transitionEndEvent, el._moveCb = function cb (e) {
            if (e && e.target !== el) {
              return
            }
            if (!e || /transform$/.test(e.propertyName)) {
              el.removeEventListener(transitionEndEvent, cb);
              el._moveCb = null;
              removeTransitionClass(el, moveClass);
            }
          });
        }
      });
    },

    methods: {
      hasMove: function hasMove (el, moveClass) {
        /* istanbul ignore if */
        if (!hasTransition) {
          return false
        }
        /* istanbul ignore if */
        if (this._hasMove) {
          return this._hasMove
        }
        // Detect whether an element with the move class applied has
        // CSS transitions. Since the element may be inside an entering
        // transition at this very moment, we make a clone of it and remove
        // all other transition classes applied to ensure only the move class
        // is applied.
        var clone = el.cloneNode();
        if (el._transitionClasses) {
          el._transitionClasses.forEach(function (cls) { removeClass(clone, cls); });
        }
        addClass(clone, moveClass);
        clone.style.display = 'none';
        this.$el.appendChild(clone);
        var info = getTransitionInfo(clone);
        this.$el.removeChild(clone);
        return (this._hasMove = info.hasTransform)
      }
    }
  };

  function callPendingCbs (c) {
    /* istanbul ignore if */
    if (c.elm._moveCb) {
      c.elm._moveCb();
    }
    /* istanbul ignore if */
    if (c.elm._enterCb) {
      c.elm._enterCb();
    }
  }

  function recordPosition (c) {
    c.data.newPos = c.elm.getBoundingClientRect();
  }

  function applyTranslation (c) {
    var oldPos = c.data.pos;
    var newPos = c.data.newPos;
    var dx = oldPos.left - newPos.left;
    var dy = oldPos.top - newPos.top;
    if (dx || dy) {
      c.data.moved = true;
      var s = c.elm.style;
      s.transform = s.WebkitTransform = "translate(" + dx + "px," + dy + "px)";
      s.transitionDuration = '0s';
    }
  }

  var platformComponents = {
    Transition: Transition,
    TransitionGroup: TransitionGroup
  };

  /*  */

  // install platform specific utils
  Vue.config.mustUseProp = mustUseProp;
  Vue.config.isReservedTag = isReservedTag;
  Vue.config.isReservedAttr = isReservedAttr;
  Vue.config.getTagNamespace = getTagNamespace;
  Vue.config.isUnknownElement = isUnknownElement;

  // install platform runtime directives & components
  extend(Vue.options.directives, platformDirectives);
  extend(Vue.options.components, platformComponents);

  // install platform patch function
  Vue.prototype.__patch__ = inBrowser ? patch : noop;

  // public mount method
  Vue.prototype.$mount = function (
    el,
    hydrating
  ) {
    el = el && inBrowser ? query(el) : undefined;
    return mountComponent(this, el, hydrating)
  };

  // devtools global hook
  /* istanbul ignore next */
  if (inBrowser) {
    setTimeout(function () {
      if (config.devtools) {
        if (devtools) {
          devtools.emit('init', Vue);
        } else {
          console[console.info ? 'info' : 'log'](
            'Download the Vue Devtools extension for a better development experience:\n' +
            'https://github.com/vuejs/vue-devtools'
          );
        }
      }
      if (config.productionTip !== false &&
        typeof console !== 'undefined'
      ) {
        console[console.info ? 'info' : 'log'](
          "You are running Vue in development mode.\n" +
          "Make sure to turn on production mode when deploying for production.\n" +
          "See more tips at https://vuejs.org/guide/deployment.html"
        );
      }
    }, 0);
  }

  const EVENT_STATUS_START = 'status:start';
  const EVENT_STATUS_UPDATE = 'status:update';
  const EVENT_STATUS_SUCCEED = 'status:success';
  const EVENT_STATUS_NOTIFY = 'status:notify';
  const EVENT_STATUS_FAIL = 'status:fail';
  const events$1 = {
    EVENT_STATUS_START,
    EVENT_STATUS_UPDATE,
    EVENT_STATUS_SUCCEED,
    EVENT_STATUS_NOTIFY,
    EVENT_STATUS_FAIL
  };
  const EventBus = new Vue();
  EventBus.$on(events$1.EVENT_STATUS_START, vm => {
    if (vm.$spinner) vm.$spinner.start();
  });
  EventBus.$on(events$1.EVENT_STATUS_UPDATE, (vm, progress) => {
    if (vm.$Progress) vm.$Progress.set(progress);
  });
  EventBus.$on(events$1.EVENT_STATUS_SUCCEED, (vm, notif) => {
    if (vm.$spinner) vm.$spinner.stop();
    if (vm.$Progress) vm.$Progress.finish();
    if (notif && notif.message && vm.$notifications) vm.$notifications.notify(notif);
  });
  EventBus.$on(events$1.EVENT_STATUS_NOTIFY, (vm, notif) => {
    if (notif && notif.message && vm.$notifications) vm.$notifications.notify(notif);
  });
  EventBus.$on(events$1.EVENT_STATUS_FAIL, (vm, notif) => {
    if (vm.$spinner) vm.$spinner.stop();
    if (vm.$Progress) vm.$Progress.fail();
    if (notif && notif.message && vm.$notifications) vm.$notifications.notify(notif);
  });

  // progress-indicator-service.js -- functions for showing progress
  var complete = 0.0; // Put this here so it's global

  function start(vm, message) {
    if (!message) {
      message = 'Starting progress';
    }

    var delay = 100;
    var stepsize = 1.0;
    complete = 0.0; // Reset this

    console.log(message);
    setTimeout(function run() {
      // Run in a delay loop
      setFunc();

      if (complete < 99) {
        setTimeout(run, delay);
      }
    }, delay);

    function setFunc() {
      complete = complete + stepsize * (1 - complete / 100); // Increase asymptotically

      EventBus.$emit(events$1.EVENT_STATUS_UPDATE, vm, complete);
    }

    EventBus.$emit(events$1.EVENT_STATUS_START, vm);
  }

  function succeed(vm, successMessage) {
    console.log(successMessage);
    complete = 100; // End the counter

    var notif = {};

    if (successMessage !== '') {
      // Success popup.
      notif = {
        message: successMessage,
        icon: 'ti-check',
        type: 'success',
        verticalAlign: 'top',
        horizontalAlign: 'right',
        timeout: 2000
      };
    }

    EventBus.$emit(events$1.EVENT_STATUS_SUCCEED, vm, notif);
  }

  function fail(vm, failMessage, error) {
    console.log(failMessage);

    if (error) {
      var msgsplit = error.message.split('Exception details:'); // WARNING, must match sc_app.py

      var usererr = msgsplit[0].replace(/\n/g, '<br>');
      console.log(error.message);
      console.log(usererr);
      var usermsg = '<b>' + failMessage + '</b>' + '<br><br>' + usererr;
    } else {
      var usermsg = '<b>' + failMessage + '</b>';
    }

    complete = 100;
    var notif = {};

    if (failMessage !== '') {
      // Put up a failure notification.
      notif = {
        message: usermsg,
        icon: 'ti-face-sad',
        type: 'warning',
        verticalAlign: 'top',
        horizontalAlign: 'right',
        timeout: 0
      };
    }

    EventBus.$emit(events$1.EVENT_STATUS_FAIL, vm, notif);
  }

  function notify(vm, notifyMessage) {
    console.log(notifyMessage);
    complete = 100; // End the counter

    var notif = {};

    if (notifyMessage !== '') {
      // Notification popup.
      notif = {
        message: notifyMessage,
        icon: 'ti-info',
        type: 'warning',
        verticalAlign: 'top',
        horizontalAlign: 'right',
        timeout: 2000
      };
    }

    EventBus.$emit(events$1.EVENT_STATUS_NOTIFY, vm, notif);
  }

  var status = {
    start,
    succeed,
    fail,
    notify
  };

  function sleep(time) {
    // Return a promise that resolves after _time_ milliseconds.
    return new Promise(resolve => setTimeout(resolve, time));
  }

  function getUniqueName(fileName, otherNames) {
    let tryName = fileName;
    let numAdded = 0;

    while (otherNames.indexOf(tryName) > -1) {
      numAdded = numAdded + 1;
      tryName = fileName + ' (' + numAdded + ')';
    }

    return tryName;
  }

  var utils = {
    sleep,
    getUniqueName
  };

  var bind$1 = function bind(fn, thisArg) {
    return function wrap() {
      var args = new Array(arguments.length);
      for (var i = 0; i < args.length; i++) {
        args[i] = arguments[i];
      }
      return fn.apply(thisArg, args);
    };
  };

  /*!
   * Determine if an object is a Buffer
   *
   * @author   Feross Aboukhadijeh <https://feross.org>
   * @license  MIT
   */

  var isBuffer = function isBuffer (obj) {
    return obj != null && obj.constructor != null &&
      typeof obj.constructor.isBuffer === 'function' && obj.constructor.isBuffer(obj)
  };

  /*global toString:true*/

  // utils is a library of generic helper functions non-specific to axios

  var toString$1 = Object.prototype.toString;

  /**
   * Determine if a value is an Array
   *
   * @param {Object} val The value to test
   * @returns {boolean} True if value is an Array, otherwise false
   */
  function isArray(val) {
    return toString$1.call(val) === '[object Array]';
  }

  /**
   * Determine if a value is an ArrayBuffer
   *
   * @param {Object} val The value to test
   * @returns {boolean} True if value is an ArrayBuffer, otherwise false
   */
  function isArrayBuffer(val) {
    return toString$1.call(val) === '[object ArrayBuffer]';
  }

  /**
   * Determine if a value is a FormData
   *
   * @param {Object} val The value to test
   * @returns {boolean} True if value is an FormData, otherwise false
   */
  function isFormData(val) {
    return (typeof FormData !== 'undefined') && (val instanceof FormData);
  }

  /**
   * Determine if a value is a view on an ArrayBuffer
   *
   * @param {Object} val The value to test
   * @returns {boolean} True if value is a view on an ArrayBuffer, otherwise false
   */
  function isArrayBufferView(val) {
    var result;
    if ((typeof ArrayBuffer !== 'undefined') && (ArrayBuffer.isView)) {
      result = ArrayBuffer.isView(val);
    } else {
      result = (val) && (val.buffer) && (val.buffer instanceof ArrayBuffer);
    }
    return result;
  }

  /**
   * Determine if a value is a String
   *
   * @param {Object} val The value to test
   * @returns {boolean} True if value is a String, otherwise false
   */
  function isString(val) {
    return typeof val === 'string';
  }

  /**
   * Determine if a value is a Number
   *
   * @param {Object} val The value to test
   * @returns {boolean} True if value is a Number, otherwise false
   */
  function isNumber(val) {
    return typeof val === 'number';
  }

  /**
   * Determine if a value is undefined
   *
   * @param {Object} val The value to test
   * @returns {boolean} True if the value is undefined, otherwise false
   */
  function isUndefined(val) {
    return typeof val === 'undefined';
  }

  /**
   * Determine if a value is an Object
   *
   * @param {Object} val The value to test
   * @returns {boolean} True if value is an Object, otherwise false
   */
  function isObject$1(val) {
    return val !== null && typeof val === 'object';
  }

  /**
   * Determine if a value is a Date
   *
   * @param {Object} val The value to test
   * @returns {boolean} True if value is a Date, otherwise false
   */
  function isDate(val) {
    return toString$1.call(val) === '[object Date]';
  }

  /**
   * Determine if a value is a File
   *
   * @param {Object} val The value to test
   * @returns {boolean} True if value is a File, otherwise false
   */
  function isFile(val) {
    return toString$1.call(val) === '[object File]';
  }

  /**
   * Determine if a value is a Blob
   *
   * @param {Object} val The value to test
   * @returns {boolean} True if value is a Blob, otherwise false
   */
  function isBlob(val) {
    return toString$1.call(val) === '[object Blob]';
  }

  /**
   * Determine if a value is a Function
   *
   * @param {Object} val The value to test
   * @returns {boolean} True if value is a Function, otherwise false
   */
  function isFunction(val) {
    return toString$1.call(val) === '[object Function]';
  }

  /**
   * Determine if a value is a Stream
   *
   * @param {Object} val The value to test
   * @returns {boolean} True if value is a Stream, otherwise false
   */
  function isStream(val) {
    return isObject$1(val) && isFunction(val.pipe);
  }

  /**
   * Determine if a value is a URLSearchParams object
   *
   * @param {Object} val The value to test
   * @returns {boolean} True if value is a URLSearchParams object, otherwise false
   */
  function isURLSearchParams(val) {
    return typeof URLSearchParams !== 'undefined' && val instanceof URLSearchParams;
  }

  /**
   * Trim excess whitespace off the beginning and end of a string
   *
   * @param {String} str The String to trim
   * @returns {String} The String freed of excess whitespace
   */
  function trim(str) {
    return str.replace(/^\s*/, '').replace(/\s*$/, '');
  }

  /**
   * Determine if we're running in a standard browser environment
   *
   * This allows axios to run in a web worker, and react-native.
   * Both environments support XMLHttpRequest, but not fully standard globals.
   *
   * web workers:
   *  typeof window -> undefined
   *  typeof document -> undefined
   *
   * react-native:
   *  navigator.product -> 'ReactNative'
   */
  function isStandardBrowserEnv() {
    if (typeof navigator !== 'undefined' && navigator.product === 'ReactNative') {
      return false;
    }
    return (
      typeof window !== 'undefined' &&
      typeof document !== 'undefined'
    );
  }

  /**
   * Iterate over an Array or an Object invoking a function for each item.
   *
   * If `obj` is an Array callback will be called passing
   * the value, index, and complete array for each item.
   *
   * If 'obj' is an Object callback will be called passing
   * the value, key, and complete object for each property.
   *
   * @param {Object|Array} obj The object to iterate
   * @param {Function} fn The callback to invoke for each item
   */
  function forEach(obj, fn) {
    // Don't bother if no value provided
    if (obj === null || typeof obj === 'undefined') {
      return;
    }

    // Force an array if not already something iterable
    if (typeof obj !== 'object') {
      /*eslint no-param-reassign:0*/
      obj = [obj];
    }

    if (isArray(obj)) {
      // Iterate over array values
      for (var i = 0, l = obj.length; i < l; i++) {
        fn.call(null, obj[i], i, obj);
      }
    } else {
      // Iterate over object keys
      for (var key in obj) {
        if (Object.prototype.hasOwnProperty.call(obj, key)) {
          fn.call(null, obj[key], key, obj);
        }
      }
    }
  }

  /**
   * Accepts varargs expecting each argument to be an object, then
   * immutably merges the properties of each object and returns result.
   *
   * When multiple objects contain the same key the later object in
   * the arguments list will take precedence.
   *
   * Example:
   *
   * ```js
   * var result = merge({foo: 123}, {foo: 456});
   * console.log(result.foo); // outputs 456
   * ```
   *
   * @param {Object} obj1 Object to merge
   * @returns {Object} Result of all merge properties
   */
  function merge(/* obj1, obj2, obj3, ... */) {
    var result = {};
    function assignValue(val, key) {
      if (typeof result[key] === 'object' && typeof val === 'object') {
        result[key] = merge(result[key], val);
      } else {
        result[key] = val;
      }
    }

    for (var i = 0, l = arguments.length; i < l; i++) {
      forEach(arguments[i], assignValue);
    }
    return result;
  }

  /**
   * Extends object a by mutably adding to it the properties of object b.
   *
   * @param {Object} a The object to be extended
   * @param {Object} b The object to copy properties from
   * @param {Object} thisArg The object to bind function to
   * @return {Object} The resulting value of object a
   */
  function extend$1(a, b, thisArg) {
    forEach(b, function assignValue(val, key) {
      if (thisArg && typeof val === 'function') {
        a[key] = bind$1(val, thisArg);
      } else {
        a[key] = val;
      }
    });
    return a;
  }

  var utils$1 = {
    isArray: isArray,
    isArrayBuffer: isArrayBuffer,
    isBuffer: isBuffer,
    isFormData: isFormData,
    isArrayBufferView: isArrayBufferView,
    isString: isString,
    isNumber: isNumber,
    isObject: isObject$1,
    isUndefined: isUndefined,
    isDate: isDate,
    isFile: isFile,
    isBlob: isBlob,
    isFunction: isFunction,
    isStream: isStream,
    isURLSearchParams: isURLSearchParams,
    isStandardBrowserEnv: isStandardBrowserEnv,
    forEach: forEach,
    merge: merge,
    extend: extend$1,
    trim: trim
  };

  var normalizeHeaderName = function normalizeHeaderName(headers, normalizedName) {
    utils$1.forEach(headers, function processHeader(value, name) {
      if (name !== normalizedName && name.toUpperCase() === normalizedName.toUpperCase()) {
        headers[normalizedName] = value;
        delete headers[name];
      }
    });
  };

  /**
   * Update an Error with the specified config, error code, and response.
   *
   * @param {Error} error The error to update.
   * @param {Object} config The config.
   * @param {string} [code] The error code (for example, 'ECONNABORTED').
   * @param {Object} [request] The request.
   * @param {Object} [response] The response.
   * @returns {Error} The error.
   */
  var enhanceError = function enhanceError(error, config, code, request, response) {
    error.config = config;
    if (code) {
      error.code = code;
    }
    error.request = request;
    error.response = response;
    return error;
  };

  /**
   * Create an Error with the specified message, config, error code, request and response.
   *
   * @param {string} message The error message.
   * @param {Object} config The config.
   * @param {string} [code] The error code (for example, 'ECONNABORTED').
   * @param {Object} [request] The request.
   * @param {Object} [response] The response.
   * @returns {Error} The created error.
   */
  var createError = function createError(message, config, code, request, response) {
    var error = new Error(message);
    return enhanceError(error, config, code, request, response);
  };

  /**
   * Resolve or reject a Promise based on response status.
   *
   * @param {Function} resolve A function that resolves the promise.
   * @param {Function} reject A function that rejects the promise.
   * @param {object} response The response.
   */
  var settle = function settle(resolve, reject, response) {
    var validateStatus = response.config.validateStatus;
    // Note: status is not exposed by XDomainRequest
    if (!response.status || !validateStatus || validateStatus(response.status)) {
      resolve(response);
    } else {
      reject(createError(
        'Request failed with status code ' + response.status,
        response.config,
        null,
        response.request,
        response
      ));
    }
  };

  function encode(val) {
    return encodeURIComponent(val).
      replace(/%40/gi, '@').
      replace(/%3A/gi, ':').
      replace(/%24/g, '$').
      replace(/%2C/gi, ',').
      replace(/%20/g, '+').
      replace(/%5B/gi, '[').
      replace(/%5D/gi, ']');
  }

  /**
   * Build a URL by appending params to the end
   *
   * @param {string} url The base of the url (e.g., http://www.google.com)
   * @param {object} [params] The params to be appended
   * @returns {string} The formatted url
   */
  var buildURL = function buildURL(url, params, paramsSerializer) {
    /*eslint no-param-reassign:0*/
    if (!params) {
      return url;
    }

    var serializedParams;
    if (paramsSerializer) {
      serializedParams = paramsSerializer(params);
    } else if (utils$1.isURLSearchParams(params)) {
      serializedParams = params.toString();
    } else {
      var parts = [];

      utils$1.forEach(params, function serialize(val, key) {
        if (val === null || typeof val === 'undefined') {
          return;
        }

        if (utils$1.isArray(val)) {
          key = key + '[]';
        } else {
          val = [val];
        }

        utils$1.forEach(val, function parseValue(v) {
          if (utils$1.isDate(v)) {
            v = v.toISOString();
          } else if (utils$1.isObject(v)) {
            v = JSON.stringify(v);
          }
          parts.push(encode(key) + '=' + encode(v));
        });
      });

      serializedParams = parts.join('&');
    }

    if (serializedParams) {
      url += (url.indexOf('?') === -1 ? '?' : '&') + serializedParams;
    }

    return url;
  };

  // Headers whose duplicates are ignored by node
  // c.f. https://nodejs.org/api/http.html#http_message_headers
  var ignoreDuplicateOf = [
    'age', 'authorization', 'content-length', 'content-type', 'etag',
    'expires', 'from', 'host', 'if-modified-since', 'if-unmodified-since',
    'last-modified', 'location', 'max-forwards', 'proxy-authorization',
    'referer', 'retry-after', 'user-agent'
  ];

  /**
   * Parse headers into an object
   *
   * ```
   * Date: Wed, 27 Aug 2014 08:58:49 GMT
   * Content-Type: application/json
   * Connection: keep-alive
   * Transfer-Encoding: chunked
   * ```
   *
   * @param {String} headers Headers needing to be parsed
   * @returns {Object} Headers parsed into an object
   */
  var parseHeaders = function parseHeaders(headers) {
    var parsed = {};
    var key;
    var val;
    var i;

    if (!headers) { return parsed; }

    utils$1.forEach(headers.split('\n'), function parser(line) {
      i = line.indexOf(':');
      key = utils$1.trim(line.substr(0, i)).toLowerCase();
      val = utils$1.trim(line.substr(i + 1));

      if (key) {
        if (parsed[key] && ignoreDuplicateOf.indexOf(key) >= 0) {
          return;
        }
        if (key === 'set-cookie') {
          parsed[key] = (parsed[key] ? parsed[key] : []).concat([val]);
        } else {
          parsed[key] = parsed[key] ? parsed[key] + ', ' + val : val;
        }
      }
    });

    return parsed;
  };

  var isURLSameOrigin = (
    utils$1.isStandardBrowserEnv() ?

    // Standard browser envs have full support of the APIs needed to test
    // whether the request URL is of the same origin as current location.
    (function standardBrowserEnv() {
      var msie = /(msie|trident)/i.test(navigator.userAgent);
      var urlParsingNode = document.createElement('a');
      var originURL;

      /**
      * Parse a URL to discover it's components
      *
      * @param {String} url The URL to be parsed
      * @returns {Object}
      */
      function resolveURL(url) {
        var href = url;

        if (msie) {
          // IE needs attribute set twice to normalize properties
          urlParsingNode.setAttribute('href', href);
          href = urlParsingNode.href;
        }

        urlParsingNode.setAttribute('href', href);

        // urlParsingNode provides the UrlUtils interface - http://url.spec.whatwg.org/#urlutils
        return {
          href: urlParsingNode.href,
          protocol: urlParsingNode.protocol ? urlParsingNode.protocol.replace(/:$/, '') : '',
          host: urlParsingNode.host,
          search: urlParsingNode.search ? urlParsingNode.search.replace(/^\?/, '') : '',
          hash: urlParsingNode.hash ? urlParsingNode.hash.replace(/^#/, '') : '',
          hostname: urlParsingNode.hostname,
          port: urlParsingNode.port,
          pathname: (urlParsingNode.pathname.charAt(0) === '/') ?
                    urlParsingNode.pathname :
                    '/' + urlParsingNode.pathname
        };
      }

      originURL = resolveURL(window.location.href);

      /**
      * Determine if a URL shares the same origin as the current location
      *
      * @param {String} requestURL The URL to test
      * @returns {boolean} True if URL shares the same origin, otherwise false
      */
      return function isURLSameOrigin(requestURL) {
        var parsed = (utils$1.isString(requestURL)) ? resolveURL(requestURL) : requestURL;
        return (parsed.protocol === originURL.protocol &&
              parsed.host === originURL.host);
      };
    })() :

    // Non standard browser envs (web workers, react-native) lack needed support.
    (function nonStandardBrowserEnv() {
      return function isURLSameOrigin() {
        return true;
      };
    })()
  );

  var cookies = (
    utils$1.isStandardBrowserEnv() ?

    // Standard browser envs support document.cookie
    (function standardBrowserEnv() {
      return {
        write: function write(name, value, expires, path, domain, secure) {
          var cookie = [];
          cookie.push(name + '=' + encodeURIComponent(value));

          if (utils$1.isNumber(expires)) {
            cookie.push('expires=' + new Date(expires).toGMTString());
          }

          if (utils$1.isString(path)) {
            cookie.push('path=' + path);
          }

          if (utils$1.isString(domain)) {
            cookie.push('domain=' + domain);
          }

          if (secure === true) {
            cookie.push('secure');
          }

          document.cookie = cookie.join('; ');
        },

        read: function read(name) {
          var match = document.cookie.match(new RegExp('(^|;\\s*)(' + name + ')=([^;]*)'));
          return (match ? decodeURIComponent(match[3]) : null);
        },

        remove: function remove(name) {
          this.write(name, '', Date.now() - 86400000);
        }
      };
    })() :

    // Non standard browser env (web workers, react-native) lack needed support.
    (function nonStandardBrowserEnv() {
      return {
        write: function write() {},
        read: function read() { return null; },
        remove: function remove() {}
      };
    })()
  );

  var xhr = function xhrAdapter(config) {
    return new Promise(function dispatchXhrRequest(resolve, reject) {
      var requestData = config.data;
      var requestHeaders = config.headers;

      if (utils$1.isFormData(requestData)) {
        delete requestHeaders['Content-Type']; // Let the browser set it
      }

      var request = new XMLHttpRequest();

      // HTTP basic authentication
      if (config.auth) {
        var username = config.auth.username || '';
        var password = config.auth.password || '';
        requestHeaders.Authorization = 'Basic ' + btoa(username + ':' + password);
      }

      request.open(config.method.toUpperCase(), buildURL(config.url, config.params, config.paramsSerializer), true);

      // Set the request timeout in MS
      request.timeout = config.timeout;

      // Listen for ready state
      request.onreadystatechange = function handleLoad() {
        if (!request || request.readyState !== 4) {
          return;
        }

        // The request errored out and we didn't get a response, this will be
        // handled by onerror instead
        // With one exception: request that using file: protocol, most browsers
        // will return status as 0 even though it's a successful request
        if (request.status === 0 && !(request.responseURL && request.responseURL.indexOf('file:') === 0)) {
          return;
        }

        // Prepare the response
        var responseHeaders = 'getAllResponseHeaders' in request ? parseHeaders(request.getAllResponseHeaders()) : null;
        var responseData = !config.responseType || config.responseType === 'text' ? request.responseText : request.response;
        var response = {
          data: responseData,
          status: request.status,
          statusText: request.statusText,
          headers: responseHeaders,
          config: config,
          request: request
        };

        settle(resolve, reject, response);

        // Clean up request
        request = null;
      };

      // Handle low level network errors
      request.onerror = function handleError() {
        // Real errors are hidden from us by the browser
        // onerror should only fire if it's a network error
        reject(createError('Network Error', config, null, request));

        // Clean up request
        request = null;
      };

      // Handle timeout
      request.ontimeout = function handleTimeout() {
        reject(createError('timeout of ' + config.timeout + 'ms exceeded', config, 'ECONNABORTED',
          request));

        // Clean up request
        request = null;
      };

      // Add xsrf header
      // This is only done if running in a standard browser environment.
      // Specifically not if we're in a web worker, or react-native.
      if (utils$1.isStandardBrowserEnv()) {
        var cookies$$1 = cookies;

        // Add xsrf header
        var xsrfValue = (config.withCredentials || isURLSameOrigin(config.url)) && config.xsrfCookieName ?
            cookies$$1.read(config.xsrfCookieName) :
            undefined;

        if (xsrfValue) {
          requestHeaders[config.xsrfHeaderName] = xsrfValue;
        }
      }

      // Add headers to the request
      if ('setRequestHeader' in request) {
        utils$1.forEach(requestHeaders, function setRequestHeader(val, key) {
          if (typeof requestData === 'undefined' && key.toLowerCase() === 'content-type') {
            // Remove Content-Type if data is undefined
            delete requestHeaders[key];
          } else {
            // Otherwise add header to the request
            request.setRequestHeader(key, val);
          }
        });
      }

      // Add withCredentials to request if needed
      if (config.withCredentials) {
        request.withCredentials = true;
      }

      // Add responseType to request if needed
      if (config.responseType) {
        try {
          request.responseType = config.responseType;
        } catch (e) {
          // Expected DOMException thrown by browsers not compatible XMLHttpRequest Level 2.
          // But, this can be suppressed for 'json' type as it can be parsed by default 'transformResponse' function.
          if (config.responseType !== 'json') {
            throw e;
          }
        }
      }

      // Handle progress if needed
      if (typeof config.onDownloadProgress === 'function') {
        request.addEventListener('progress', config.onDownloadProgress);
      }

      // Not all browsers support upload events
      if (typeof config.onUploadProgress === 'function' && request.upload) {
        request.upload.addEventListener('progress', config.onUploadProgress);
      }

      if (config.cancelToken) {
        // Handle cancellation
        config.cancelToken.promise.then(function onCanceled(cancel) {
          if (!request) {
            return;
          }

          request.abort();
          reject(cancel);
          // Clean up request
          request = null;
        });
      }

      if (requestData === undefined) {
        requestData = null;
      }

      // Send the request
      request.send(requestData);
    });
  };

  var DEFAULT_CONTENT_TYPE = {
    'Content-Type': 'application/x-www-form-urlencoded'
  };

  function setContentTypeIfUnset(headers, value) {
    if (!utils$1.isUndefined(headers) && utils$1.isUndefined(headers['Content-Type'])) {
      headers['Content-Type'] = value;
    }
  }

  function getDefaultAdapter() {
    var adapter;
    if (typeof XMLHttpRequest !== 'undefined') {
      // For browsers use XHR adapter
      adapter = xhr;
    } else if (typeof process !== 'undefined') {
      // For node use HTTP adapter
      adapter = xhr;
    }
    return adapter;
  }

  var defaults = {
    adapter: getDefaultAdapter(),

    transformRequest: [function transformRequest(data, headers) {
      normalizeHeaderName(headers, 'Content-Type');
      if (utils$1.isFormData(data) ||
        utils$1.isArrayBuffer(data) ||
        utils$1.isBuffer(data) ||
        utils$1.isStream(data) ||
        utils$1.isFile(data) ||
        utils$1.isBlob(data)
      ) {
        return data;
      }
      if (utils$1.isArrayBufferView(data)) {
        return data.buffer;
      }
      if (utils$1.isURLSearchParams(data)) {
        setContentTypeIfUnset(headers, 'application/x-www-form-urlencoded;charset=utf-8');
        return data.toString();
      }
      if (utils$1.isObject(data)) {
        setContentTypeIfUnset(headers, 'application/json;charset=utf-8');
        return JSON.stringify(data);
      }
      return data;
    }],

    transformResponse: [function transformResponse(data) {
      /*eslint no-param-reassign:0*/
      if (typeof data === 'string') {
        try {
          data = JSON.parse(data);
        } catch (e) { /* Ignore */ }
      }
      return data;
    }],

    /**
     * A timeout in milliseconds to abort a request. If set to 0 (default) a
     * timeout is not created.
     */
    timeout: 0,

    xsrfCookieName: 'XSRF-TOKEN',
    xsrfHeaderName: 'X-XSRF-TOKEN',

    maxContentLength: -1,

    validateStatus: function validateStatus(status) {
      return status >= 200 && status < 300;
    }
  };

  defaults.headers = {
    common: {
      'Accept': 'application/json, text/plain, */*'
    }
  };

  utils$1.forEach(['delete', 'get', 'head'], function forEachMethodNoData(method) {
    defaults.headers[method] = {};
  });

  utils$1.forEach(['post', 'put', 'patch'], function forEachMethodWithData(method) {
    defaults.headers[method] = utils$1.merge(DEFAULT_CONTENT_TYPE);
  });

  var defaults_1 = defaults;

  function InterceptorManager() {
    this.handlers = [];
  }

  /**
   * Add a new interceptor to the stack
   *
   * @param {Function} fulfilled The function to handle `then` for a `Promise`
   * @param {Function} rejected The function to handle `reject` for a `Promise`
   *
   * @return {Number} An ID used to remove interceptor later
   */
  InterceptorManager.prototype.use = function use(fulfilled, rejected) {
    this.handlers.push({
      fulfilled: fulfilled,
      rejected: rejected
    });
    return this.handlers.length - 1;
  };

  /**
   * Remove an interceptor from the stack
   *
   * @param {Number} id The ID that was returned by `use`
   */
  InterceptorManager.prototype.eject = function eject(id) {
    if (this.handlers[id]) {
      this.handlers[id] = null;
    }
  };

  /**
   * Iterate over all the registered interceptors
   *
   * This method is particularly useful for skipping over any
   * interceptors that may have become `null` calling `eject`.
   *
   * @param {Function} fn The function to call for each interceptor
   */
  InterceptorManager.prototype.forEach = function forEach(fn) {
    utils$1.forEach(this.handlers, function forEachHandler(h) {
      if (h !== null) {
        fn(h);
      }
    });
  };

  var InterceptorManager_1 = InterceptorManager;

  /**
   * Transform the data for a request or a response
   *
   * @param {Object|String} data The data to be transformed
   * @param {Array} headers The headers for the request or response
   * @param {Array|Function} fns A single function or Array of functions
   * @returns {*} The resulting transformed data
   */
  var transformData = function transformData(data, headers, fns) {
    /*eslint no-param-reassign:0*/
    utils$1.forEach(fns, function transform(fn) {
      data = fn(data, headers);
    });

    return data;
  };

  var isCancel = function isCancel(value) {
    return !!(value && value.__CANCEL__);
  };

  /**
   * Determines whether the specified URL is absolute
   *
   * @param {string} url The URL to test
   * @returns {boolean} True if the specified URL is absolute, otherwise false
   */
  var isAbsoluteURL = function isAbsoluteURL(url) {
    // A URL is considered absolute if it begins with "<scheme>://" or "//" (protocol-relative URL).
    // RFC 3986 defines scheme name as a sequence of characters beginning with a letter and followed
    // by any combination of letters, digits, plus, period, or hyphen.
    return /^([a-z][a-z\d\+\-\.]*:)?\/\//i.test(url);
  };

  /**
   * Creates a new URL by combining the specified URLs
   *
   * @param {string} baseURL The base URL
   * @param {string} relativeURL The relative URL
   * @returns {string} The combined URL
   */
  var combineURLs = function combineURLs(baseURL, relativeURL) {
    return relativeURL
      ? baseURL.replace(/\/+$/, '') + '/' + relativeURL.replace(/^\/+/, '')
      : baseURL;
  };

  /**
   * Throws a `Cancel` if cancellation has been requested.
   */
  function throwIfCancellationRequested(config) {
    if (config.cancelToken) {
      config.cancelToken.throwIfRequested();
    }
  }

  /**
   * Dispatch a request to the server using the configured adapter.
   *
   * @param {object} config The config that is to be used for the request
   * @returns {Promise} The Promise to be fulfilled
   */
  var dispatchRequest = function dispatchRequest(config) {
    throwIfCancellationRequested(config);

    // Support baseURL config
    if (config.baseURL && !isAbsoluteURL(config.url)) {
      config.url = combineURLs(config.baseURL, config.url);
    }

    // Ensure headers exist
    config.headers = config.headers || {};

    // Transform request data
    config.data = transformData(
      config.data,
      config.headers,
      config.transformRequest
    );

    // Flatten headers
    config.headers = utils$1.merge(
      config.headers.common || {},
      config.headers[config.method] || {},
      config.headers || {}
    );

    utils$1.forEach(
      ['delete', 'get', 'head', 'post', 'put', 'patch', 'common'],
      function cleanHeaderConfig(method) {
        delete config.headers[method];
      }
    );

    var adapter = config.adapter || defaults_1.adapter;

    return adapter(config).then(function onAdapterResolution(response) {
      throwIfCancellationRequested(config);

      // Transform response data
      response.data = transformData(
        response.data,
        response.headers,
        config.transformResponse
      );

      return response;
    }, function onAdapterRejection(reason) {
      if (!isCancel(reason)) {
        throwIfCancellationRequested(config);

        // Transform response data
        if (reason && reason.response) {
          reason.response.data = transformData(
            reason.response.data,
            reason.response.headers,
            config.transformResponse
          );
        }
      }

      return Promise.reject(reason);
    });
  };

  /**
   * Create a new instance of Axios
   *
   * @param {Object} instanceConfig The default config for the instance
   */
  function Axios(instanceConfig) {
    this.defaults = instanceConfig;
    this.interceptors = {
      request: new InterceptorManager_1(),
      response: new InterceptorManager_1()
    };
  }

  /**
   * Dispatch a request
   *
   * @param {Object} config The config specific for this request (merged with this.defaults)
   */
  Axios.prototype.request = function request(config) {
    /*eslint no-param-reassign:0*/
    // Allow for axios('example/url'[, config]) a la fetch API
    if (typeof config === 'string') {
      config = utils$1.merge({
        url: arguments[0]
      }, arguments[1]);
    }

    config = utils$1.merge(defaults_1, {method: 'get'}, this.defaults, config);
    config.method = config.method.toLowerCase();

    // Hook up interceptors middleware
    var chain = [dispatchRequest, undefined];
    var promise = Promise.resolve(config);

    this.interceptors.request.forEach(function unshiftRequestInterceptors(interceptor) {
      chain.unshift(interceptor.fulfilled, interceptor.rejected);
    });

    this.interceptors.response.forEach(function pushResponseInterceptors(interceptor) {
      chain.push(interceptor.fulfilled, interceptor.rejected);
    });

    while (chain.length) {
      promise = promise.then(chain.shift(), chain.shift());
    }

    return promise;
  };

  // Provide aliases for supported request methods
  utils$1.forEach(['delete', 'get', 'head', 'options'], function forEachMethodNoData(method) {
    /*eslint func-names:0*/
    Axios.prototype[method] = function(url, config) {
      return this.request(utils$1.merge(config || {}, {
        method: method,
        url: url
      }));
    };
  });

  utils$1.forEach(['post', 'put', 'patch'], function forEachMethodWithData(method) {
    /*eslint func-names:0*/
    Axios.prototype[method] = function(url, data, config) {
      return this.request(utils$1.merge(config || {}, {
        method: method,
        url: url,
        data: data
      }));
    };
  });

  var Axios_1 = Axios;

  /**
   * A `Cancel` is an object that is thrown when an operation is canceled.
   *
   * @class
   * @param {string=} message The message.
   */
  function Cancel(message) {
    this.message = message;
  }

  Cancel.prototype.toString = function toString() {
    return 'Cancel' + (this.message ? ': ' + this.message : '');
  };

  Cancel.prototype.__CANCEL__ = true;

  var Cancel_1 = Cancel;

  /**
   * A `CancelToken` is an object that can be used to request cancellation of an operation.
   *
   * @class
   * @param {Function} executor The executor function.
   */
  function CancelToken(executor) {
    if (typeof executor !== 'function') {
      throw new TypeError('executor must be a function.');
    }

    var resolvePromise;
    this.promise = new Promise(function promiseExecutor(resolve) {
      resolvePromise = resolve;
    });

    var token = this;
    executor(function cancel(message) {
      if (token.reason) {
        // Cancellation has already been requested
        return;
      }

      token.reason = new Cancel_1(message);
      resolvePromise(token.reason);
    });
  }

  /**
   * Throws a `Cancel` if cancellation has been requested.
   */
  CancelToken.prototype.throwIfRequested = function throwIfRequested() {
    if (this.reason) {
      throw this.reason;
    }
  };

  /**
   * Returns an object that contains a new `CancelToken` and a function that, when called,
   * cancels the `CancelToken`.
   */
  CancelToken.source = function source() {
    var cancel;
    var token = new CancelToken(function executor(c) {
      cancel = c;
    });
    return {
      token: token,
      cancel: cancel
    };
  };

  var CancelToken_1 = CancelToken;

  /**
   * Syntactic sugar for invoking a function and expanding an array for arguments.
   *
   * Common use case would be to use `Function.prototype.apply`.
   *
   *  ```js
   *  function f(x, y, z) {}
   *  var args = [1, 2, 3];
   *  f.apply(null, args);
   *  ```
   *
   * With `spread` this example can be re-written.
   *
   *  ```js
   *  spread(function(x, y, z) {})([1, 2, 3]);
   *  ```
   *
   * @param {Function} callback
   * @returns {Function}
   */
  var spread = function spread(callback) {
    return function wrap(arr) {
      return callback.apply(null, arr);
    };
  };

  /**
   * Create an instance of Axios
   *
   * @param {Object} defaultConfig The default config for the instance
   * @return {Axios} A new instance of Axios
   */
  function createInstance(defaultConfig) {
    var context = new Axios_1(defaultConfig);
    var instance = bind$1(Axios_1.prototype.request, context);

    // Copy axios.prototype to instance
    utils$1.extend(instance, Axios_1.prototype, context);

    // Copy context to instance
    utils$1.extend(instance, context);

    return instance;
  }

  // Create the default instance to be exported
  var axios = createInstance(defaults_1);

  // Expose Axios class to allow class inheritance
  axios.Axios = Axios_1;

  // Factory for creating new instances
  axios.create = function create(instanceConfig) {
    return createInstance(utils$1.merge(defaults_1, instanceConfig));
  };

  // Expose Cancel & CancelToken
  axios.Cancel = Cancel_1;
  axios.CancelToken = CancelToken_1;
  axios.isCancel = isCancel;

  // Expose all/spread
  axios.all = function all(promises) {
    return Promise.all(promises);
  };
  axios.spread = spread;

  var axios_1 = axios;

  // Allow use of default import syntax in TypeScript
  var default_1 = axios;
  axios_1.default = default_1;

  var axios$1 = axios_1;

  var commonjsGlobal = typeof globalThis !== 'undefined' ? globalThis : typeof window !== 'undefined' ? window : typeof global !== 'undefined' ? global : typeof self !== 'undefined' ? self : {};

  function unwrapExports (x) {
  	return x && x.__esModule && Object.prototype.hasOwnProperty.call(x, 'default') ? x['default'] : x;
  }

  function createCommonjsModule(fn, module) {
  	return module = { exports: {} }, fn(module, module.exports), module.exports;
  }

  var FileSaver_min = createCommonjsModule(function (module, exports) {
  (function(a,b){b();})(commonjsGlobal,function(){function b(a,b){return "undefined"==typeof b?b={autoBom:!1}:"object"!=typeof b&&(console.warn("Deprecated: Expected third argument to be a object"),b={autoBom:!b}),b.autoBom&&/^\s*(?:text\/\S*|application\/xml|\S*\/\S*\+xml)\s*;.*charset\s*=\s*utf-8/i.test(a.type)?new Blob(["\uFEFF",a],{type:a.type}):a}function c(b,c,d){var e=new XMLHttpRequest;e.open("GET",b),e.responseType="blob",e.onload=function(){a(e.response,c,d);},e.onerror=function(){console.error("could not download file");},e.send();}function d(a){var b=new XMLHttpRequest;b.open("HEAD",a,!1);try{b.send();}catch(a){}return 200<=b.status&&299>=b.status}function e(a){try{a.dispatchEvent(new MouseEvent("click"));}catch(c){var b=document.createEvent("MouseEvents");b.initMouseEvent("click",!0,!0,window,0,0,0,80,20,!1,!1,!1,!1,0,null),a.dispatchEvent(b);}}var f="object"==typeof window&&window.window===window?window:"object"==typeof self&&self.self===self?self:"object"==typeof commonjsGlobal&&commonjsGlobal.global===commonjsGlobal?commonjsGlobal:void 0,a=f.saveAs||("object"!=typeof window||window!==f?function(){}:"download"in HTMLAnchorElement.prototype?function(b,g,h){var i=f.URL||f.webkitURL,j=document.createElement("a");g=g||b.name||"download",j.download=g,j.rel="noopener","string"==typeof b?(j.href=b,j.origin===location.origin?e(j):d(j.href)?c(b,g,h):e(j,j.target="_blank")):(j.href=i.createObjectURL(b),setTimeout(function(){i.revokeObjectURL(j.href);},4E4),setTimeout(function(){e(j);},0));}:"msSaveOrOpenBlob"in navigator?function(f,g,h){if(g=g||f.name||"download","string"!=typeof f)navigator.msSaveOrOpenBlob(b(f,h),g);else if(d(f))c(f,g,h);else{var i=document.createElement("a");i.href=f,i.target="_blank",setTimeout(function(){e(i);});}}:function(a,b,d,e){if(e=e||open("","_blank"),e&&(e.document.title=e.document.body.innerText="downloading..."),"string"==typeof a)return c(a,b,d);var g="application/octet-stream"===a.type,h=/constructor/i.test(f.HTMLElement)||f.safari,i=/CriOS\/[\d]+/.test(navigator.userAgent);if((i||g&&h)&&"object"==typeof FileReader){var j=new FileReader;j.onloadend=function(){var a=j.result;a=i?a:a.replace(/^data:[^;]*;/,"data:attachment/file;"),e?e.location.href=a:location=a,e=null;},j.readAsDataURL(a);}else{var k=f.URL||f.webkitURL,l=k.createObjectURL(a);e?e.location=l:location.href=l,e=null,setTimeout(function(){k.revokeObjectURL(l);},4E4);}});f.saveAs=a.saveAs=a,module.exports=a;});


  });

  // rpc-service.js -- RPC functions for Vue to call

  function consoleLogCommand(type, funcname, args, kwargs) {
    if (!args) {
      // Don't show any arguments if none are passed in.
      args = '';
    }

    if (!kwargs) {
      // Don't show any kwargs if none are passed in.
      kwargs = '';
    }

    console.log("RPC service call (" + type + "): " + funcname, args, kwargs);
  } // readJsonFromBlob(theBlob) -- Attempt to convert a Blob passed in to a JSON. Passes back a Promise.


  function readJsonFromBlob(theBlob) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader(); // Create a FileReader; reader.result contains the contents of blob as text when this is called

      reader.addEventListener("loadend", function () {
        // Create a callback for after the load attempt is finished
        try {
          // Call a resolve passing back a JSON version of this.
          var jsonresult = JSON.parse(reader.result); // Try the conversion.

          resolve(jsonresult); // (Assuming successful) make the Promise resolve with the JSON result.
        } catch (e) {
          reject(Error('Failed to convert blob to JSON')); // On failure to convert to JSON, reject the Promise.
        }
      });
      reader.readAsText(theBlob); // Start the load attempt, trying to read the blob in as text.
    });
  }

  var rpcs = {
    rpc(funcname, args, kwargs) {
      // rpc() -- normalRPC() /api/procedure calls in api.py.
      consoleLogCommand("normal", funcname, args, kwargs); // Log the RPC call.

      return new Promise((resolve, reject) => {
        // Do the RPC processing, returning results as a Promise.
        axios$1.post('/api/rpcs', {
          // Send the POST request for the RPC call.
          funcname: funcname,
          args: args,
          kwargs: kwargs
        }).then(response => {
          if (typeof response.data.error !== 'undefined') {
            // If there is an error in the POST response.
            console.log('RPC error: ' + response.data.error);
            reject(Error(response.data.error));
          } else {
            console.log('RPC succeeded');
            resolve(response); // Signal success with the response.
          }
        }).catch(error => {
          console.log('RPC error: ' + error);

          if (error.response) {
            // If there was an actual response returned from the server...
            if (typeof error.response.data.exception !== 'undefined') {
              // If we have exception information in the response (which indicates an exception on the server side)...
              reject(Error(error.response.data.exception)); // For now, reject with an error message matching the exception.
            }
          } else {
            reject(error); // Reject with the error axios got.
          }
        });
      });
    },

    download(funcname, args, kwargs) {
      // download() -- download() /api/download calls in api.py.
      consoleLogCommand("download", funcname, args, kwargs); // Log the download RPC call.

      return new Promise((resolve, reject) => {
        // Do the RPC processing, returning results as a Promise.
        axios$1.post('/api/rpcs', {
          // Send the POST request for the RPC call.
          funcname: funcname,
          args: args,
          kwargs: kwargs
        }, {
          responseType: 'blob'
        }).then(response => {
          readJsonFromBlob(response.data).then(responsedata => {
            if (typeof responsedata.error != 'undefined') {
              // If we have error information in the response (which indicates a logical error on the server side)...
              reject(Error(responsedata.error)); // For now, reject with an error message matching the error.
            }
          }).catch(error2 => {
            // An error here indicates we do in fact have a file to download.
            var blob = new Blob([response.data]); // Create a new blob object (containing the file data) from the response.data component.

            var filename = response.headers.filename; // Grab the file name from response.headers.

            FileSaver_min(blob, filename); // Bring up the browser dialog allowing the user to save the file or cancel doing so.

            resolve(response); // Signal success with the response.
          });
        }).catch(error => {
          if (error.response) {
            // If there was an actual response returned from the server...
            readJsonFromBlob(error.response.data).then(responsedata => {
              if (typeof responsedata.exception !== 'undefined') {
                // If we have exception information in the response (which indicates an exception on the server side)...
                reject(Error(responsedata.exception)); // For now, reject with an error message matching the exception.
              }
            }).catch(error2 => {
              reject(error); // Reject with the error axios got.
            });
          } else {
            reject(error); // Otherwise (no response was delivered), reject with the error axios got.
          }
        });
      });
    },

    // upload() -- upload() /api/upload calls in api.py.
    upload(funcname, args, kwargs, fileType) {
      consoleLogCommand("upload", funcname, args, kwargs); // Log the upload RPC call.

      return new Promise((resolve, reject) => {
        // Do the RPC processing, returning results as a Promise.
        var onFileChange = e => {
          // Function for trapping the change event that has the user-selected file.
          var files = e.target.files || e.dataTransfer.files; // Pull out the files (should only be 1) that were selected.

          if (!files.length) // If no files were selected, reject the promise.
            reject(Error('No file selected'));
          const formData = new FormData(); // Create a FormData object for holding the file.

          formData.append('uploadfile', files[0]); // Put the selected file in the formData object with 'uploadfile' key.

          formData.append('funcname', funcname); // Add the RPC function name to the form data.

          formData.append('args', JSON.stringify(args)); // Add args and kwargs to the form data.

          formData.append('kwargs', JSON.stringify(kwargs));
          axios$1.post('/api/rpcs', formData) // Use a POST request to pass along file to the server.
          .then(response => {
            // If there is an error in the POST response.
            if (typeof response.data.error != 'undefined') {
              reject(Error(response.data.error));
            }

            resolve(response); // Signal success with the response.
          }).catch(error => {
            if (error.response) {
              // If there was an actual response returned from the server...
              if (typeof error.response.data.exception != 'undefined') {
                // If we have exception information in the response (which indicates an exception on the server side)...
                reject(Error(error.response.data.exception)); // For now, reject with an error message matching the exception.
              }
            }

            reject(error); // Reject with the error axios got.
          });
        }; // Create an invisible file input element and set its change callback to our onFileChange function.


        var inElem = document.createElement('input');
        inElem.setAttribute('type', 'file');
        inElem.setAttribute('accept', fileType);
        inElem.addEventListener('change', onFileChange);
        inElem.click(); // Manually click the button to open the file dialog.
      });
    }

  };

  /*
   * Graphing functions (shared between calibration, scenarios, and optimization)
   */
  let mpld3 = null;

  if (typeof d3 !== 'undefined') {
    mpld3 = require('mpld3');
  }

  function placeholders(vm, startVal) {
    let indices = [];

    if (!startVal) {
      startVal = 0;
    }

    for (let i = startVal; i <= 100; i++) {
      indices.push(i);
      vm.showGraphDivs.push(false);
      vm.showLegendDivs.push(false);
    }

    return indices;
  }

  function clearGraphs(vm) {
    for (let index = 0; index <= 100; index++) {
      let divlabel = 'fig' + index;
      let div = document.getElementById(divlabel); // CK: Not sure if this is necessary? To ensure the div is clear first

      while (div && div.firstChild) {
        div.removeChild(div.firstChild);
      }

      vm.hasGraphs = false;
    }
  }

  function makeGraphs(vm, data, routepath) {
    if (typeof d3 === 'undefined') {
      console.log("please include d3 to use the makeGraphs function");
      return false;
    }

    if (routepath && routepath !== vm.$route.path) {
      // Don't render graphs if we've changed page
      console.log('Not rendering graphs since route changed: ' + routepath + ' vs. ' + vm.$route.path);
    } else {
      // Proceed...
      let waitingtime = 0.5;
      var graphdata = data.graphs; // var legenddata = data.legends

      status.start(vm); // Start indicating progress.

      vm.hasGraphs = true;
      utils.sleep(waitingtime * 1000).then(response => {
        let n_plots = graphdata.length; // let n_legends = legenddata.length

        console.log('Rendering ' + n_plots + ' graphs'); // if (n_plots !== n_legends) {
        //   console.log('WARNING: different numbers of plots and legends: ' + n_plots + ' vs. ' + n_legends)
        // }

        for (var index = 0; index <= n_plots; index++) {
          console.log('Rendering plot ' + index);
          var figlabel = 'fig' + index;
          var figdiv = document.getElementById(figlabel); // CK: Not sure if this is necessary? To ensure the div is clear first

          if (figdiv) {
            while (figdiv.firstChild) {
              figdiv.removeChild(figdiv.firstChild);
            }
          } else {
            console.log('WARNING: figdiv not found: ' + figlabel);
          } // Show figure containers


          if (index >= 1 && index < n_plots) {
            var figcontainerlabel = 'figcontainer' + index;
            var figcontainerdiv = document.getElementById(figcontainerlabel); // CK: Not sure if this is necessary? To ensure the div is clear first

            if (figcontainerdiv) {
              figcontainerdiv.style.display = 'flex';
            } else {
              console.log('WARNING: figcontainerdiv not found: ' + figcontainerlabel);
            } // var legendlabel = 'legend' + index
            // var legenddiv  = document.getElementById(legendlabel);
            // if (legenddiv) {
            //   while (legenddiv.firstChild) {
            //     legenddiv.removeChild(legenddiv.firstChild);
            //   }
            // } else {
            //   console.log('WARNING: legenddiv not found: ' + legendlabel)
            // }

          } // Draw figures


          try {
            mpld3.draw_figure(figlabel, graphdata[index], function (fig, element) {
              fig.setXTicks(6, function (d) {
                return d3.format('.0f')(d);
              }); // fig.setYTicks(null, function (d) { // Looks too weird with 500m for 0.5
              //   return d3.format('.2s')(d);
              // });
            }, true);
          } catch (error) {
            console.log('Could not plot graph: ' + error.message);
          } // Draw legends
          // if (index>=1 && index<n_plots) {
          //   try {
          //     mpld3.draw_figure(legendlabel, legenddata[index], function (fig, element) {
          //     });
          //   } catch (error) {
          //     console.log(error)
          //   }
          //
          // }


          vm.showGraphDivs[index] = true;
        }

        status.succeed(vm, 'Graphs created'); // CK: This should be a promise, otherwise this appears before the graphs do
      });
    }
  } //
  // Graphs DOM functions
  //


  function showBrowserWindowSize() {
    let w = window.innerWidth;
    let h = window.innerHeight;
    let ow = window.outerWidth; //including toolbars and status bar etc.

    let oh = window.outerHeight;
    console.log('Browser window size:');
    console.log(w, h, ow, oh);
  }

  function scaleElem(svg, frac) {
    // It might ultimately be better to redraw the graph, but this works
    let width = svg.getAttribute("width");
    let height = svg.getAttribute("height");
    let viewBox = svg.getAttribute("viewBox");

    if (!viewBox) {
      svg.setAttribute("viewBox", '0 0 ' + width + ' ' + height);
    } // if this causes the image to look weird, you may want to look at "preserveAspectRatio" attribute


    svg.setAttribute("width", width * frac);
    svg.setAttribute("height", height * frac);
  }

  function scaleFigs(vm, frac) {
    vm.figscale = vm.figscale * frac;

    if (frac === 1.0) {
      frac = 1.0 / vm.figscale;
      vm.figscale = 1.0;
    }

    let graphs = window.top.document.querySelectorAll('svg.mpld3-figure');

    for (let g = 0; g < graphs.length; g++) {
      scaleElem(graphs[g], frac);
    }
  } //
  // Legend functions
  // 


  function addListener(vm) {
    document.addEventListener('mousemove', function (e) {
      onMouseUpdate(e, vm);
    }, false);
  }

  function onMouseUpdate(e, vm) {
    vm.mousex = e.pageX;
    vm.mousey = e.pageY; // console.log(vm.mousex, vm.mousey)
  }

  function createDialogs(vm) {
    let vals = placeholders(vm);

    for (let val in vals) {
      newDialog(vm, val, 'Dialog ' + val, 'Placeholder content ' + val);
    }
  } // Create a new dialog


  function newDialog(vm, id, name, content) {
    let options = {
      left: 123 + Number(id),
      top: 123
    };
    let style = {
      options: options
    };
    let properties = {
      id,
      name,
      content,
      style,
      options
    };
    return vm.openDialogs.push(properties);
  }

  function findDialog(vm, id, dialogs) {
    console.log('looking');
    let index = dialogs.findIndex(val => {
      return String(val.id) === String(id); // Force type conversion
    });
    return index > -1 ? index : null;
  } // "Show" the dialog


  function maximize(vm, id) {
    let index = Number(id);
    let DDlabel = 'DD' + id; // DD for dialog-drag

    let DDdiv = document.getElementById(DDlabel);

    if (DDdiv) {
      DDdiv.style.left = String(vm.mousex - 80) + 'px';
      DDdiv.style.top = String(vm.mousey - 300) + 'px';
    } else {
      console.log('WARNING: DDdiv not found: ' + DDlabel);
    }

    if (index !== null) {
      vm.openDialogs[index].options.left = vm.mousex - 80; // Before opening, move it to where the mouse currently is

      vm.openDialogs[index].options.top = vm.mousey - 300;
    }

    vm.showLegendDivs[index] = true; // Not really used, but here for completeness

    let containerlabel = 'legendcontainer' + id;
    let containerdiv = document.getElementById(containerlabel);

    if (containerdiv) {
      containerdiv.style.display = 'inline-block'; // Ensure they're invisible
    } else {
      console.log('WARNING: containerdiv not found: ' + containerlabel);
    }
  } // "Hide" the dialog


  function minimize(vm, id) {
    let index = Number(id);
    vm.showLegendDivs[index] = false;
    let containerlabel = 'legendcontainer' + id;
    let containerdiv = document.getElementById(containerlabel);

    if (containerdiv) {
      containerdiv.style.display = 'none'; // Ensure they're invisible
    } else {
      console.log('WARNING: containerdiv not found: ' + containerlabel);
    }
  }

  var graphs = {
    placeholders,
    clearGraphs,
    makeGraphs,
    scaleFigs,
    showBrowserWindowSize,
    addListener,
    onMouseUpdate,
    createDialogs,
    newDialog,
    findDialog,
    maximize,
    minimize,
    mpld3
  };

  // task-service.js -- task queuing functions for Vue to call
  // sec.), and a remote task function name and its args, try to launch 
  // the task, then wait for the waiting time, then try to get the 
  // result.

  function getTaskResultWaiting(task_id, waitingtime, func_name, args, kwargs) {
    if (!args) {
      // Set the arguments to an empty list if none are passed in.
      args = [];
    }

    return new Promise((resolve, reject) => {
      rpcs.rpc('launch_task', [task_id, func_name, args, kwargs]) // Launch the task.
      .then(response => {
        utils.sleep(waitingtime * 1000) // Sleep waitingtime seconds.
        .then(response2 => {
          rpcs.rpc('get_task_result', [task_id]) // Get the result of the task.
          .then(response3 => {
            rpcs.rpc('delete_task', [task_id]); // Clean up the task_id task.

            resolve(response3); // Signal success with the result response.
          }).catch(error => {
            // While we might want to clean up the task as below, the Celery
            // worker is likely to "resurrect" the task if it actually is
            // running the task to completion.
            // Clean up the task_id task.
            // rpcCall('delete_task', [task_id])
            reject(error); // Reject with the error the task result get attempt gave.
          });
        });
      }).catch(error => {
        reject(error); // Reject with the error the launch gave.
      });
    });
  } // getTaskResultPolling() -- given a task_id string, a timeout time (in 
  // sec.), a polling interval (also in sec.), and a remote task function name
  //  and its args, try to launch the task, then start the polling if this is 
  // successful, returning the ultimate results of the polling process. 


  function getTaskResultPolling(task_id, timeout, pollinterval, func_name, args, kwargs) {
    if (!args) {
      // Set the arguments to an empty list if none are passed in.
      args = [];
    }

    return new Promise((resolve, reject) => {
      rpcs.rpc('launch_task', [task_id, func_name, args, kwargs]) // Launch the task.
      .then(response => {
        pollStep(task_id, timeout, pollinterval, 0) // Do the whole sequence of polling steps, starting with the first (recursive) call.
        .then(response2 => {
          resolve(response2); // Resolve with the final polling result.
        }).catch(error => {
          reject(error); // Reject with the error the polling gave.
        });
      }).catch(error => {
        reject(error); // Reject with the error the launch gave.
      });
    });
  } // pollStep() -- A polling step for getTaskResultPolling().  Uses the task_id, 
  // a timeout value (in sec.) a poll interval (in sec.) and the time elapsed 
  // since the start of the entire polling process.  If timeout is zero or 
  // negative, no timeout check is applied.  Otherwise, an error will be 
  // returned if the polling has gone on beyond the timeout period.  Otherwise, 
  // this function does a sleep() and then a check_task().  If the task is 
  // completed, it will get the result.  Otherwise, it will recursively spawn 
  // another pollStep().


  function pollStep(task_id, timeout, pollinterval, elapsedtime) {
    return new Promise((resolve, reject) => {
      if (elapsedtime > timeout && timeout > 0) {
        // Check to see if the elapsed time is longer than the timeout (and we have a timeout we actually want to check against) and if so, fail.
        reject(Error('Task polling timed out'));
      } else {
        // Otherwise, we've not run out of time yet, so do a polling step.
        utils.sleep(pollinterval * 1000) // Sleep timeout seconds.
        .then(response => {
          rpcs.rpc('check_task', [task_id]) // Check the status of the task.
          .then(response2 => {
            if (response2.data.task.status == 'completed') {
              // If the task is completed...
              rpcs.rpc('get_task_result', [task_id]) // Get the result of the task.
              .then(response3 => {
                rpcs.rpc('delete_task', [task_id]); // Clean up the task_id task.

                resolve(response3); // Signal success with the response.
              }).catch(error => {
                reject(error); // Reject with the error the task result get attempt gave.
              });
            } else if (response2.data.task.status == 'error') {
              // Otherwise, if the task ended in an error...
              reject(Error(response2.data.task.errorText)); // Reject with an error for the exception.
            } else {
              // Otherwise, do another poll step, passing in an incremented elapsed time.
              pollStep(task_id, timeout, pollinterval, elapsedtime + pollinterval).then(response3 => {
                resolve(response3); // Resolve with the result of the next polling step (which may include subsequent (recursive) steps.
              });
            }
          });
        });
      }
    });
  }

  var tasks = {
    getTaskResultWaiting,
    getTaskResultPolling
  };

  var core = createCommonjsModule(function (module, exports) {
  (function (root, factory) {
  	{
  		// CommonJS
  		module.exports = exports = factory();
  	}
  }(commonjsGlobal, function () {

  	/**
  	 * CryptoJS core components.
  	 */
  	var CryptoJS = CryptoJS || (function (Math, undefined) {
  	    /*
  	     * Local polyfil of Object.create
  	     */
  	    var create = Object.create || (function () {
  	        function F() {}
  	        return function (obj) {
  	            var subtype;

  	            F.prototype = obj;

  	            subtype = new F();

  	            F.prototype = null;

  	            return subtype;
  	        };
  	    }());

  	    /**
  	     * CryptoJS namespace.
  	     */
  	    var C = {};

  	    /**
  	     * Library namespace.
  	     */
  	    var C_lib = C.lib = {};

  	    /**
  	     * Base object for prototypal inheritance.
  	     */
  	    var Base = C_lib.Base = (function () {


  	        return {
  	            /**
  	             * Creates a new object that inherits from this object.
  	             *
  	             * @param {Object} overrides Properties to copy into the new object.
  	             *
  	             * @return {Object} The new object.
  	             *
  	             * @static
  	             *
  	             * @example
  	             *
  	             *     var MyType = CryptoJS.lib.Base.extend({
  	             *         field: 'value',
  	             *
  	             *         method: function () {
  	             *         }
  	             *     });
  	             */
  	            extend: function (overrides) {
  	                // Spawn
  	                var subtype = create(this);

  	                // Augment
  	                if (overrides) {
  	                    subtype.mixIn(overrides);
  	                }

  	                // Create default initializer
  	                if (!subtype.hasOwnProperty('init') || this.init === subtype.init) {
  	                    subtype.init = function () {
  	                        subtype.$super.init.apply(this, arguments);
  	                    };
  	                }

  	                // Initializer's prototype is the subtype object
  	                subtype.init.prototype = subtype;

  	                // Reference supertype
  	                subtype.$super = this;

  	                return subtype;
  	            },

  	            /**
  	             * Extends this object and runs the init method.
  	             * Arguments to create() will be passed to init().
  	             *
  	             * @return {Object} The new object.
  	             *
  	             * @static
  	             *
  	             * @example
  	             *
  	             *     var instance = MyType.create();
  	             */
  	            create: function () {
  	                var instance = this.extend();
  	                instance.init.apply(instance, arguments);

  	                return instance;
  	            },

  	            /**
  	             * Initializes a newly created object.
  	             * Override this method to add some logic when your objects are created.
  	             *
  	             * @example
  	             *
  	             *     var MyType = CryptoJS.lib.Base.extend({
  	             *         init: function () {
  	             *             // ...
  	             *         }
  	             *     });
  	             */
  	            init: function () {
  	            },

  	            /**
  	             * Copies properties into this object.
  	             *
  	             * @param {Object} properties The properties to mix in.
  	             *
  	             * @example
  	             *
  	             *     MyType.mixIn({
  	             *         field: 'value'
  	             *     });
  	             */
  	            mixIn: function (properties) {
  	                for (var propertyName in properties) {
  	                    if (properties.hasOwnProperty(propertyName)) {
  	                        this[propertyName] = properties[propertyName];
  	                    }
  	                }

  	                // IE won't copy toString using the loop above
  	                if (properties.hasOwnProperty('toString')) {
  	                    this.toString = properties.toString;
  	                }
  	            },

  	            /**
  	             * Creates a copy of this object.
  	             *
  	             * @return {Object} The clone.
  	             *
  	             * @example
  	             *
  	             *     var clone = instance.clone();
  	             */
  	            clone: function () {
  	                return this.init.prototype.extend(this);
  	            }
  	        };
  	    }());

  	    /**
  	     * An array of 32-bit words.
  	     *
  	     * @property {Array} words The array of 32-bit words.
  	     * @property {number} sigBytes The number of significant bytes in this word array.
  	     */
  	    var WordArray = C_lib.WordArray = Base.extend({
  	        /**
  	         * Initializes a newly created word array.
  	         *
  	         * @param {Array} words (Optional) An array of 32-bit words.
  	         * @param {number} sigBytes (Optional) The number of significant bytes in the words.
  	         *
  	         * @example
  	         *
  	         *     var wordArray = CryptoJS.lib.WordArray.create();
  	         *     var wordArray = CryptoJS.lib.WordArray.create([0x00010203, 0x04050607]);
  	         *     var wordArray = CryptoJS.lib.WordArray.create([0x00010203, 0x04050607], 6);
  	         */
  	        init: function (words, sigBytes) {
  	            words = this.words = words || [];

  	            if (sigBytes != undefined) {
  	                this.sigBytes = sigBytes;
  	            } else {
  	                this.sigBytes = words.length * 4;
  	            }
  	        },

  	        /**
  	         * Converts this word array to a string.
  	         *
  	         * @param {Encoder} encoder (Optional) The encoding strategy to use. Default: CryptoJS.enc.Hex
  	         *
  	         * @return {string} The stringified word array.
  	         *
  	         * @example
  	         *
  	         *     var string = wordArray + '';
  	         *     var string = wordArray.toString();
  	         *     var string = wordArray.toString(CryptoJS.enc.Utf8);
  	         */
  	        toString: function (encoder) {
  	            return (encoder || Hex).stringify(this);
  	        },

  	        /**
  	         * Concatenates a word array to this word array.
  	         *
  	         * @param {WordArray} wordArray The word array to append.
  	         *
  	         * @return {WordArray} This word array.
  	         *
  	         * @example
  	         *
  	         *     wordArray1.concat(wordArray2);
  	         */
  	        concat: function (wordArray) {
  	            // Shortcuts
  	            var thisWords = this.words;
  	            var thatWords = wordArray.words;
  	            var thisSigBytes = this.sigBytes;
  	            var thatSigBytes = wordArray.sigBytes;

  	            // Clamp excess bits
  	            this.clamp();

  	            // Concat
  	            if (thisSigBytes % 4) {
  	                // Copy one byte at a time
  	                for (var i = 0; i < thatSigBytes; i++) {
  	                    var thatByte = (thatWords[i >>> 2] >>> (24 - (i % 4) * 8)) & 0xff;
  	                    thisWords[(thisSigBytes + i) >>> 2] |= thatByte << (24 - ((thisSigBytes + i) % 4) * 8);
  	                }
  	            } else {
  	                // Copy one word at a time
  	                for (var i = 0; i < thatSigBytes; i += 4) {
  	                    thisWords[(thisSigBytes + i) >>> 2] = thatWords[i >>> 2];
  	                }
  	            }
  	            this.sigBytes += thatSigBytes;

  	            // Chainable
  	            return this;
  	        },

  	        /**
  	         * Removes insignificant bits.
  	         *
  	         * @example
  	         *
  	         *     wordArray.clamp();
  	         */
  	        clamp: function () {
  	            // Shortcuts
  	            var words = this.words;
  	            var sigBytes = this.sigBytes;

  	            // Clamp
  	            words[sigBytes >>> 2] &= 0xffffffff << (32 - (sigBytes % 4) * 8);
  	            words.length = Math.ceil(sigBytes / 4);
  	        },

  	        /**
  	         * Creates a copy of this word array.
  	         *
  	         * @return {WordArray} The clone.
  	         *
  	         * @example
  	         *
  	         *     var clone = wordArray.clone();
  	         */
  	        clone: function () {
  	            var clone = Base.clone.call(this);
  	            clone.words = this.words.slice(0);

  	            return clone;
  	        },

  	        /**
  	         * Creates a word array filled with random bytes.
  	         *
  	         * @param {number} nBytes The number of random bytes to generate.
  	         *
  	         * @return {WordArray} The random word array.
  	         *
  	         * @static
  	         *
  	         * @example
  	         *
  	         *     var wordArray = CryptoJS.lib.WordArray.random(16);
  	         */
  	        random: function (nBytes) {
  	            var words = [];

  	            var r = (function (m_w) {
  	                var m_w = m_w;
  	                var m_z = 0x3ade68b1;
  	                var mask = 0xffffffff;

  	                return function () {
  	                    m_z = (0x9069 * (m_z & 0xFFFF) + (m_z >> 0x10)) & mask;
  	                    m_w = (0x4650 * (m_w & 0xFFFF) + (m_w >> 0x10)) & mask;
  	                    var result = ((m_z << 0x10) + m_w) & mask;
  	                    result /= 0x100000000;
  	                    result += 0.5;
  	                    return result * (Math.random() > .5 ? 1 : -1);
  	                }
  	            });

  	            for (var i = 0, rcache; i < nBytes; i += 4) {
  	                var _r = r((rcache || Math.random()) * 0x100000000);

  	                rcache = _r() * 0x3ade67b7;
  	                words.push((_r() * 0x100000000) | 0);
  	            }

  	            return new WordArray.init(words, nBytes);
  	        }
  	    });

  	    /**
  	     * Encoder namespace.
  	     */
  	    var C_enc = C.enc = {};

  	    /**
  	     * Hex encoding strategy.
  	     */
  	    var Hex = C_enc.Hex = {
  	        /**
  	         * Converts a word array to a hex string.
  	         *
  	         * @param {WordArray} wordArray The word array.
  	         *
  	         * @return {string} The hex string.
  	         *
  	         * @static
  	         *
  	         * @example
  	         *
  	         *     var hexString = CryptoJS.enc.Hex.stringify(wordArray);
  	         */
  	        stringify: function (wordArray) {
  	            // Shortcuts
  	            var words = wordArray.words;
  	            var sigBytes = wordArray.sigBytes;

  	            // Convert
  	            var hexChars = [];
  	            for (var i = 0; i < sigBytes; i++) {
  	                var bite = (words[i >>> 2] >>> (24 - (i % 4) * 8)) & 0xff;
  	                hexChars.push((bite >>> 4).toString(16));
  	                hexChars.push((bite & 0x0f).toString(16));
  	            }

  	            return hexChars.join('');
  	        },

  	        /**
  	         * Converts a hex string to a word array.
  	         *
  	         * @param {string} hexStr The hex string.
  	         *
  	         * @return {WordArray} The word array.
  	         *
  	         * @static
  	         *
  	         * @example
  	         *
  	         *     var wordArray = CryptoJS.enc.Hex.parse(hexString);
  	         */
  	        parse: function (hexStr) {
  	            // Shortcut
  	            var hexStrLength = hexStr.length;

  	            // Convert
  	            var words = [];
  	            for (var i = 0; i < hexStrLength; i += 2) {
  	                words[i >>> 3] |= parseInt(hexStr.substr(i, 2), 16) << (24 - (i % 8) * 4);
  	            }

  	            return new WordArray.init(words, hexStrLength / 2);
  	        }
  	    };

  	    /**
  	     * Latin1 encoding strategy.
  	     */
  	    var Latin1 = C_enc.Latin1 = {
  	        /**
  	         * Converts a word array to a Latin1 string.
  	         *
  	         * @param {WordArray} wordArray The word array.
  	         *
  	         * @return {string} The Latin1 string.
  	         *
  	         * @static
  	         *
  	         * @example
  	         *
  	         *     var latin1String = CryptoJS.enc.Latin1.stringify(wordArray);
  	         */
  	        stringify: function (wordArray) {
  	            // Shortcuts
  	            var words = wordArray.words;
  	            var sigBytes = wordArray.sigBytes;

  	            // Convert
  	            var latin1Chars = [];
  	            for (var i = 0; i < sigBytes; i++) {
  	                var bite = (words[i >>> 2] >>> (24 - (i % 4) * 8)) & 0xff;
  	                latin1Chars.push(String.fromCharCode(bite));
  	            }

  	            return latin1Chars.join('');
  	        },

  	        /**
  	         * Converts a Latin1 string to a word array.
  	         *
  	         * @param {string} latin1Str The Latin1 string.
  	         *
  	         * @return {WordArray} The word array.
  	         *
  	         * @static
  	         *
  	         * @example
  	         *
  	         *     var wordArray = CryptoJS.enc.Latin1.parse(latin1String);
  	         */
  	        parse: function (latin1Str) {
  	            // Shortcut
  	            var latin1StrLength = latin1Str.length;

  	            // Convert
  	            var words = [];
  	            for (var i = 0; i < latin1StrLength; i++) {
  	                words[i >>> 2] |= (latin1Str.charCodeAt(i) & 0xff) << (24 - (i % 4) * 8);
  	            }

  	            return new WordArray.init(words, latin1StrLength);
  	        }
  	    };

  	    /**
  	     * UTF-8 encoding strategy.
  	     */
  	    var Utf8 = C_enc.Utf8 = {
  	        /**
  	         * Converts a word array to a UTF-8 string.
  	         *
  	         * @param {WordArray} wordArray The word array.
  	         *
  	         * @return {string} The UTF-8 string.
  	         *
  	         * @static
  	         *
  	         * @example
  	         *
  	         *     var utf8String = CryptoJS.enc.Utf8.stringify(wordArray);
  	         */
  	        stringify: function (wordArray) {
  	            try {
  	                return decodeURIComponent(escape(Latin1.stringify(wordArray)));
  	            } catch (e) {
  	                throw new Error('Malformed UTF-8 data');
  	            }
  	        },

  	        /**
  	         * Converts a UTF-8 string to a word array.
  	         *
  	         * @param {string} utf8Str The UTF-8 string.
  	         *
  	         * @return {WordArray} The word array.
  	         *
  	         * @static
  	         *
  	         * @example
  	         *
  	         *     var wordArray = CryptoJS.enc.Utf8.parse(utf8String);
  	         */
  	        parse: function (utf8Str) {
  	            return Latin1.parse(unescape(encodeURIComponent(utf8Str)));
  	        }
  	    };

  	    /**
  	     * Abstract buffered block algorithm template.
  	     *
  	     * The property blockSize must be implemented in a concrete subtype.
  	     *
  	     * @property {number} _minBufferSize The number of blocks that should be kept unprocessed in the buffer. Default: 0
  	     */
  	    var BufferedBlockAlgorithm = C_lib.BufferedBlockAlgorithm = Base.extend({
  	        /**
  	         * Resets this block algorithm's data buffer to its initial state.
  	         *
  	         * @example
  	         *
  	         *     bufferedBlockAlgorithm.reset();
  	         */
  	        reset: function () {
  	            // Initial values
  	            this._data = new WordArray.init();
  	            this._nDataBytes = 0;
  	        },

  	        /**
  	         * Adds new data to this block algorithm's buffer.
  	         *
  	         * @param {WordArray|string} data The data to append. Strings are converted to a WordArray using UTF-8.
  	         *
  	         * @example
  	         *
  	         *     bufferedBlockAlgorithm._append('data');
  	         *     bufferedBlockAlgorithm._append(wordArray);
  	         */
  	        _append: function (data) {
  	            // Convert string to WordArray, else assume WordArray already
  	            if (typeof data == 'string') {
  	                data = Utf8.parse(data);
  	            }

  	            // Append
  	            this._data.concat(data);
  	            this._nDataBytes += data.sigBytes;
  	        },

  	        /**
  	         * Processes available data blocks.
  	         *
  	         * This method invokes _doProcessBlock(offset), which must be implemented by a concrete subtype.
  	         *
  	         * @param {boolean} doFlush Whether all blocks and partial blocks should be processed.
  	         *
  	         * @return {WordArray} The processed data.
  	         *
  	         * @example
  	         *
  	         *     var processedData = bufferedBlockAlgorithm._process();
  	         *     var processedData = bufferedBlockAlgorithm._process(!!'flush');
  	         */
  	        _process: function (doFlush) {
  	            // Shortcuts
  	            var data = this._data;
  	            var dataWords = data.words;
  	            var dataSigBytes = data.sigBytes;
  	            var blockSize = this.blockSize;
  	            var blockSizeBytes = blockSize * 4;

  	            // Count blocks ready
  	            var nBlocksReady = dataSigBytes / blockSizeBytes;
  	            if (doFlush) {
  	                // Round up to include partial blocks
  	                nBlocksReady = Math.ceil(nBlocksReady);
  	            } else {
  	                // Round down to include only full blocks,
  	                // less the number of blocks that must remain in the buffer
  	                nBlocksReady = Math.max((nBlocksReady | 0) - this._minBufferSize, 0);
  	            }

  	            // Count words ready
  	            var nWordsReady = nBlocksReady * blockSize;

  	            // Count bytes ready
  	            var nBytesReady = Math.min(nWordsReady * 4, dataSigBytes);

  	            // Process blocks
  	            if (nWordsReady) {
  	                for (var offset = 0; offset < nWordsReady; offset += blockSize) {
  	                    // Perform concrete-algorithm logic
  	                    this._doProcessBlock(dataWords, offset);
  	                }

  	                // Remove processed words
  	                var processedWords = dataWords.splice(0, nWordsReady);
  	                data.sigBytes -= nBytesReady;
  	            }

  	            // Return processed words
  	            return new WordArray.init(processedWords, nBytesReady);
  	        },

  	        /**
  	         * Creates a copy of this object.
  	         *
  	         * @return {Object} The clone.
  	         *
  	         * @example
  	         *
  	         *     var clone = bufferedBlockAlgorithm.clone();
  	         */
  	        clone: function () {
  	            var clone = Base.clone.call(this);
  	            clone._data = this._data.clone();

  	            return clone;
  	        },

  	        _minBufferSize: 0
  	    });

  	    /**
  	     * Abstract hasher template.
  	     *
  	     * @property {number} blockSize The number of 32-bit words this hasher operates on. Default: 16 (512 bits)
  	     */
  	    var Hasher = C_lib.Hasher = BufferedBlockAlgorithm.extend({
  	        /**
  	         * Configuration options.
  	         */
  	        cfg: Base.extend(),

  	        /**
  	         * Initializes a newly created hasher.
  	         *
  	         * @param {Object} cfg (Optional) The configuration options to use for this hash computation.
  	         *
  	         * @example
  	         *
  	         *     var hasher = CryptoJS.algo.SHA256.create();
  	         */
  	        init: function (cfg) {
  	            // Apply config defaults
  	            this.cfg = this.cfg.extend(cfg);

  	            // Set initial values
  	            this.reset();
  	        },

  	        /**
  	         * Resets this hasher to its initial state.
  	         *
  	         * @example
  	         *
  	         *     hasher.reset();
  	         */
  	        reset: function () {
  	            // Reset data buffer
  	            BufferedBlockAlgorithm.reset.call(this);

  	            // Perform concrete-hasher logic
  	            this._doReset();
  	        },

  	        /**
  	         * Updates this hasher with a message.
  	         *
  	         * @param {WordArray|string} messageUpdate The message to append.
  	         *
  	         * @return {Hasher} This hasher.
  	         *
  	         * @example
  	         *
  	         *     hasher.update('message');
  	         *     hasher.update(wordArray);
  	         */
  	        update: function (messageUpdate) {
  	            // Append
  	            this._append(messageUpdate);

  	            // Update the hash
  	            this._process();

  	            // Chainable
  	            return this;
  	        },

  	        /**
  	         * Finalizes the hash computation.
  	         * Note that the finalize operation is effectively a destructive, read-once operation.
  	         *
  	         * @param {WordArray|string} messageUpdate (Optional) A final message update.
  	         *
  	         * @return {WordArray} The hash.
  	         *
  	         * @example
  	         *
  	         *     var hash = hasher.finalize();
  	         *     var hash = hasher.finalize('message');
  	         *     var hash = hasher.finalize(wordArray);
  	         */
  	        finalize: function (messageUpdate) {
  	            // Final message update
  	            if (messageUpdate) {
  	                this._append(messageUpdate);
  	            }

  	            // Perform concrete-hasher logic
  	            var hash = this._doFinalize();

  	            return hash;
  	        },

  	        blockSize: 512/32,

  	        /**
  	         * Creates a shortcut function to a hasher's object interface.
  	         *
  	         * @param {Hasher} hasher The hasher to create a helper for.
  	         *
  	         * @return {Function} The shortcut function.
  	         *
  	         * @static
  	         *
  	         * @example
  	         *
  	         *     var SHA256 = CryptoJS.lib.Hasher._createHelper(CryptoJS.algo.SHA256);
  	         */
  	        _createHelper: function (hasher) {
  	            return function (message, cfg) {
  	                return new hasher.init(cfg).finalize(message);
  	            };
  	        },

  	        /**
  	         * Creates a shortcut function to the HMAC's object interface.
  	         *
  	         * @param {Hasher} hasher The hasher to use in this HMAC helper.
  	         *
  	         * @return {Function} The shortcut function.
  	         *
  	         * @static
  	         *
  	         * @example
  	         *
  	         *     var HmacSHA256 = CryptoJS.lib.Hasher._createHmacHelper(CryptoJS.algo.SHA256);
  	         */
  	        _createHmacHelper: function (hasher) {
  	            return function (message, key) {
  	                return new C_algo.HMAC.init(hasher, key).finalize(message);
  	            };
  	        }
  	    });

  	    /**
  	     * Algorithm namespace.
  	     */
  	    var C_algo = C.algo = {};

  	    return C;
  	}(Math));


  	return CryptoJS;

  }));
  });

  var sha256 = createCommonjsModule(function (module, exports) {
  (function (root, factory) {
  	{
  		// CommonJS
  		module.exports = exports = factory(core);
  	}
  }(commonjsGlobal, function (CryptoJS) {

  	(function (Math) {
  	    // Shortcuts
  	    var C = CryptoJS;
  	    var C_lib = C.lib;
  	    var WordArray = C_lib.WordArray;
  	    var Hasher = C_lib.Hasher;
  	    var C_algo = C.algo;

  	    // Initialization and round constants tables
  	    var H = [];
  	    var K = [];

  	    // Compute constants
  	    (function () {
  	        function isPrime(n) {
  	            var sqrtN = Math.sqrt(n);
  	            for (var factor = 2; factor <= sqrtN; factor++) {
  	                if (!(n % factor)) {
  	                    return false;
  	                }
  	            }

  	            return true;
  	        }

  	        function getFractionalBits(n) {
  	            return ((n - (n | 0)) * 0x100000000) | 0;
  	        }

  	        var n = 2;
  	        var nPrime = 0;
  	        while (nPrime < 64) {
  	            if (isPrime(n)) {
  	                if (nPrime < 8) {
  	                    H[nPrime] = getFractionalBits(Math.pow(n, 1 / 2));
  	                }
  	                K[nPrime] = getFractionalBits(Math.pow(n, 1 / 3));

  	                nPrime++;
  	            }

  	            n++;
  	        }
  	    }());

  	    // Reusable object
  	    var W = [];

  	    /**
  	     * SHA-256 hash algorithm.
  	     */
  	    var SHA256 = C_algo.SHA256 = Hasher.extend({
  	        _doReset: function () {
  	            this._hash = new WordArray.init(H.slice(0));
  	        },

  	        _doProcessBlock: function (M, offset) {
  	            // Shortcut
  	            var H = this._hash.words;

  	            // Working variables
  	            var a = H[0];
  	            var b = H[1];
  	            var c = H[2];
  	            var d = H[3];
  	            var e = H[4];
  	            var f = H[5];
  	            var g = H[6];
  	            var h = H[7];

  	            // Computation
  	            for (var i = 0; i < 64; i++) {
  	                if (i < 16) {
  	                    W[i] = M[offset + i] | 0;
  	                } else {
  	                    var gamma0x = W[i - 15];
  	                    var gamma0  = ((gamma0x << 25) | (gamma0x >>> 7))  ^
  	                                  ((gamma0x << 14) | (gamma0x >>> 18)) ^
  	                                   (gamma0x >>> 3);

  	                    var gamma1x = W[i - 2];
  	                    var gamma1  = ((gamma1x << 15) | (gamma1x >>> 17)) ^
  	                                  ((gamma1x << 13) | (gamma1x >>> 19)) ^
  	                                   (gamma1x >>> 10);

  	                    W[i] = gamma0 + W[i - 7] + gamma1 + W[i - 16];
  	                }

  	                var ch  = (e & f) ^ (~e & g);
  	                var maj = (a & b) ^ (a & c) ^ (b & c);

  	                var sigma0 = ((a << 30) | (a >>> 2)) ^ ((a << 19) | (a >>> 13)) ^ ((a << 10) | (a >>> 22));
  	                var sigma1 = ((e << 26) | (e >>> 6)) ^ ((e << 21) | (e >>> 11)) ^ ((e << 7)  | (e >>> 25));

  	                var t1 = h + sigma1 + ch + K[i] + W[i];
  	                var t2 = sigma0 + maj;

  	                h = g;
  	                g = f;
  	                f = e;
  	                e = (d + t1) | 0;
  	                d = c;
  	                c = b;
  	                b = a;
  	                a = (t1 + t2) | 0;
  	            }

  	            // Intermediate hash value
  	            H[0] = (H[0] + a) | 0;
  	            H[1] = (H[1] + b) | 0;
  	            H[2] = (H[2] + c) | 0;
  	            H[3] = (H[3] + d) | 0;
  	            H[4] = (H[4] + e) | 0;
  	            H[5] = (H[5] + f) | 0;
  	            H[6] = (H[6] + g) | 0;
  	            H[7] = (H[7] + h) | 0;
  	        },

  	        _doFinalize: function () {
  	            // Shortcuts
  	            var data = this._data;
  	            var dataWords = data.words;

  	            var nBitsTotal = this._nDataBytes * 8;
  	            var nBitsLeft = data.sigBytes * 8;

  	            // Add padding
  	            dataWords[nBitsLeft >>> 5] |= 0x80 << (24 - nBitsLeft % 32);
  	            dataWords[(((nBitsLeft + 64) >>> 9) << 4) + 14] = Math.floor(nBitsTotal / 0x100000000);
  	            dataWords[(((nBitsLeft + 64) >>> 9) << 4) + 15] = nBitsTotal;
  	            data.sigBytes = dataWords.length * 4;

  	            // Hash final blocks
  	            this._process();

  	            // Return final computed hash
  	            return this._hash;
  	        },

  	        clone: function () {
  	            var clone = Hasher.clone.call(this);
  	            clone._hash = this._hash.clone();

  	            return clone;
  	        }
  	    });

  	    /**
  	     * Shortcut function to the hasher's object interface.
  	     *
  	     * @param {WordArray|string} message The message to hash.
  	     *
  	     * @return {WordArray} The hash.
  	     *
  	     * @static
  	     *
  	     * @example
  	     *
  	     *     var hash = CryptoJS.SHA256('message');
  	     *     var hash = CryptoJS.SHA256(wordArray);
  	     */
  	    C.SHA256 = Hasher._createHelper(SHA256);

  	    /**
  	     * Shortcut function to the HMAC's object interface.
  	     *
  	     * @param {WordArray|string} message The message to hash.
  	     * @param {WordArray|string} key The secret key.
  	     *
  	     * @return {WordArray} The HMAC.
  	     *
  	     * @static
  	     *
  	     * @example
  	     *
  	     *     var hmac = CryptoJS.HmacSHA256(message, key);
  	     */
  	    C.HmacSHA256 = Hasher._createHmacHelper(SHA256);
  	}(Math));


  	return CryptoJS.SHA256;

  }));
  });

  var sha224 = createCommonjsModule(function (module, exports) {
  (function (root, factory, undef) {
  	{
  		// CommonJS
  		module.exports = exports = factory(core, sha256);
  	}
  }(commonjsGlobal, function (CryptoJS) {

  	(function () {
  	    // Shortcuts
  	    var C = CryptoJS;
  	    var C_lib = C.lib;
  	    var WordArray = C_lib.WordArray;
  	    var C_algo = C.algo;
  	    var SHA256 = C_algo.SHA256;

  	    /**
  	     * SHA-224 hash algorithm.
  	     */
  	    var SHA224 = C_algo.SHA224 = SHA256.extend({
  	        _doReset: function () {
  	            this._hash = new WordArray.init([
  	                0xc1059ed8, 0x367cd507, 0x3070dd17, 0xf70e5939,
  	                0xffc00b31, 0x68581511, 0x64f98fa7, 0xbefa4fa4
  	            ]);
  	        },

  	        _doFinalize: function () {
  	            var hash = SHA256._doFinalize.call(this);

  	            hash.sigBytes -= 4;

  	            return hash;
  	        }
  	    });

  	    /**
  	     * Shortcut function to the hasher's object interface.
  	     *
  	     * @param {WordArray|string} message The message to hash.
  	     *
  	     * @return {WordArray} The hash.
  	     *
  	     * @static
  	     *
  	     * @example
  	     *
  	     *     var hash = CryptoJS.SHA224('message');
  	     *     var hash = CryptoJS.SHA224(wordArray);
  	     */
  	    C.SHA224 = SHA256._createHelper(SHA224);

  	    /**
  	     * Shortcut function to the HMAC's object interface.
  	     *
  	     * @param {WordArray|string} message The message to hash.
  	     * @param {WordArray|string} key The secret key.
  	     *
  	     * @return {WordArray} The HMAC.
  	     *
  	     * @static
  	     *
  	     * @example
  	     *
  	     *     var hmac = CryptoJS.HmacSHA224(message, key);
  	     */
  	    C.HmacSHA224 = SHA256._createHmacHelper(SHA224);
  	}());


  	return CryptoJS.SHA224;

  }));
  });

  // loginCall() -- Call rpc() for performing a login.

  function loginCall(username, password) {
    // Get a hex version of a hashed password using the SHA224 algorithm.
    var hashPassword = sha224(password).toString(); // Make the actual RPC call.

    return rpcs.rpc('user_login', [username, hashPassword]);
  } // logoutCall() -- Call rpc() for performing a logout.


  function logoutCall() {
    // Make the actual RPC call.
    return rpcs.rpc('user_logout');
  } // getCurrentUserInfo() -- Call rpc() for reading the currently
  // logged in user.


  function getCurrentUserInfo() {
    // Make the actual RPC call.
    return rpcs.rpc('get_current_user_info');
  } // registerUser() -- Call rpc() for registering a new user.


  function registerUser(username, password, displayname, email) {
    // Get a hex version of a hashed password using the SHA224 algorithm.
    var hashPassword = sha224(password).toString(); // Make the actual RPC call.

    return rpcs.rpc('user_register', [username, hashPassword, displayname, email]);
  } // changeUserInfo() -- Call rpc() for changing a user's info.


  function changeUserInfo(username, password, displayname, email) {
    // Get a hex version of a hashed password using the SHA224 algorithm.
    var hashPassword = sha224(password).toString(); // Make the actual RPC call.

    return rpcs.rpc('user_change_info', [username, hashPassword, displayname, email]);
  } // changeUserPassword() -- Call rpc() for changing a user's password.


  function changeUserPassword(oldpassword, newpassword) {
    // Get a hex version of the hashed passwords using the SHA224 algorithm.
    var hashOldPassword = sha224(oldpassword).toString();
    var hashNewPassword = sha224(newpassword).toString(); // Make the actual RPC call.

    return rpcs.rpc('user_change_password', [hashOldPassword, hashNewPassword]);
  } // adminGetUserInfo() -- Call rpc() for getting user information at the admin level.


  function adminGetUserInfo(username) {
    // Make the actual RPC call.
    return rpcs.rpc('admin_get_user_info', [username]);
  } // deleteUser() -- Call rpc() for deleting a user.


  function deleteUser(username) {
    // Make the actual RPC call.
    return rpcs.rpc('admin_delete_user', [username]);
  } // activateUserAccount() -- Call rpc() for activating a user account.


  function activateUserAccount(username) {
    // Make the actual RPC call.
    return rpcs.rpc('admin_activate_account', [username]);
  } // deactivateUserAccount() -- Call rpc() for deactivating a user account.


  function deactivateUserAccount(username) {
    // Make the actual RPC call.
    return rpcs.rpc('admin_deactivate_account', [username]);
  } // grantUserAdminRights() -- Call rpc() for granting a user admin rights.


  function grantUserAdminRights(username) {
    // Make the actual RPC call.
    return rpcs.rpc('admin_grant_admin', [username]);
  } // revokeUserAdminRights() -- Call rpc() for revoking user admin rights.


  function revokeUserAdminRights(username) {
    // Make the actual RPC call.
    return rpcs.rpc('admin_revoke_admin', [username]);
  } // resetUserPassword() -- Call rpc() for resetting a user's password.


  function resetUserPassword(username) {
    // Make the actual RPC call.
    return rpcs.rpc('admin_reset_password', [username]);
  } // Higher level user functions that call the lower level ones above


  function getUserInfo(store) {
    // Do the actual RPC call.
    getCurrentUserInfo().then(response => {
      // Set the username to what the server indicates.
      store.commit('newUser', response.data.user);
    }).catch(error => {
      // Set the username to {}.  An error probably means the
      // user is not logged in.
      store.commit('newUser', {});
    });
  }

  function checkLoggedIn() {
    if (this.currentUser.displayname === undefined) return false;else return true;
  }

  function checkAdminLoggedIn() {
    console.log(this);

    if (this.checkLoggedIn()) {
      return this.currentUser.admin;
    }
  }

  var user = {
    loginCall,
    logoutCall,
    getCurrentUserInfo,
    registerUser,
    changeUserInfo,
    changeUserPassword,
    adminGetUserInfo,
    deleteUser,
    activateUserAccount,
    deactivateUserAccount,
    grantUserAdminRights,
    revokeUserAdminRights,
    resetUserPassword,
    getUserInfo,
    checkLoggedIn,
    checkAdminLoggedIn
  };

  var vueProgressbar = createCommonjsModule(function (module, exports) {
  !function(t,o){module.exports=o();}(commonjsGlobal,function(){!function(){if("undefined"!=typeof document){var t=document.head||document.getElementsByTagName("head")[0],o=document.createElement("style"),i=" .__cov-progress { opacity: 1; z-index: 999999; } ";o.type="text/css",o.styleSheet?o.styleSheet.cssText=i:o.appendChild(document.createTextNode(i)),t.appendChild(o);}}();var t="undefined"!=typeof window,r={render:function(){var t=this,o=t.$createElement;return (t._self._c||o)("div",{staticClass:"__cov-progress",style:t.style})},staticRenderFns:[],name:"VueProgress",serverCacheKey:function(){return "Progress"},computed:{style:function(){var t=this.progress,o=t.options,i=!!o.show,e=o.location,s={"background-color":o.canSuccess?o.color:o.failedColor,opacity:o.show?1:0,position:o.position};return "top"===e||"bottom"===e?("top"===e?s.top="0px":s.bottom="0px",o.inverse?s.right="0px":s.left="0px",s.width=t.percent+"%",s.height=o.thickness,s.transition=(i?"width "+o.transition.speed+", ":"")+"opacity "+o.transition.opacity):"left"!==e&&"right"!==e||("left"===e?s.left="0px":s.right="0px",o.inverse?s.top="0px":s.bottom="0px",s.height=t.percent+"%",s.width=o.thickness,s.transition=(i?"height "+o.transition.speed+", ":"")+"opacity "+o.transition.opacity),s},progress:function(){return t?window.VueProgressBarEventBus.RADON_LOADING_BAR:{percent:0,options:{canSuccess:!0,show:!1,color:"rgb(19, 91, 55)",failedColor:"red",thickness:"2px",transition:{speed:"0.2s",opacity:"0.6s",termination:300},location:"top",autoRevert:!0,inverse:!1}}}}};return {install:function(o){var t=1<arguments.length&&void 0!==arguments[1]?arguments[1]:{},i=(o.version.split(".")[0],"undefined"!=typeof window),e={$vm:null,state:{tFailColor:"",tColor:"",timer:null,cut:0},init:function(t){this.$vm=t;},start:function(t){var o=this;this.$vm&&(t||(t=3e3),this.$vm.RADON_LOADING_BAR.percent=0,this.$vm.RADON_LOADING_BAR.options.show=!0,this.$vm.RADON_LOADING_BAR.options.canSuccess=!0,this.state.cut=1e4/Math.floor(t),clearInterval(this.state.timer),this.state.timer=setInterval(function(){o.increase(o.state.cut*Math.random()),95<o.$vm.RADON_LOADING_BAR.percent&&o.$vm.RADON_LOADING_BAR.options.autoFinish&&o.finish();},100));},set:function(t){this.$vm.RADON_LOADING_BAR.options.show=!0,this.$vm.RADON_LOADING_BAR.options.canSuccess=!0,this.$vm.RADON_LOADING_BAR.percent=Math.floor(t);},get:function(){return Math.floor(this.$vm.RADON_LOADING_BAR.percent)},increase:function(t){this.$vm.RADON_LOADING_BAR.percent=Math.min(99,this.$vm.RADON_LOADING_BAR.percent+Math.floor(t));},decrease:function(t){this.$vm.RADON_LOADING_BAR.percent=this.$vm.RADON_LOADING_BAR.percent-Math.floor(t);},hide:function(){var t=this;clearInterval(this.state.timer),this.state.timer=null,setTimeout(function(){t.$vm.RADON_LOADING_BAR.options.show=!1,o.nextTick(function(){setTimeout(function(){t.$vm.RADON_LOADING_BAR.percent=0;},100),t.$vm.RADON_LOADING_BAR.options.autoRevert&&setTimeout(function(){t.revert();},300);});},this.$vm.RADON_LOADING_BAR.options.transition.termination);},pause:function(){clearInterval(this.state.timer);},finish:function(){this.$vm&&(this.$vm.RADON_LOADING_BAR.percent=100,this.hide());},fail:function(){this.$vm.RADON_LOADING_BAR.options.canSuccess=!1,this.$vm.RADON_LOADING_BAR.percent=100,this.hide();},setFailColor:function(t){this.$vm.RADON_LOADING_BAR.options.failedColor=t;},setColor:function(t){this.$vm.RADON_LOADING_BAR.options.color=t;},setLocation:function(t){this.$vm.RADON_LOADING_BAR.options.location=t;},setTransition:function(t){this.$vm.RADON_LOADING_BAR.options.transition=t;},tempFailColor:function(t){this.state.tFailColor=this.$vm.RADON_LOADING_BAR.options.failedColor,this.$vm.RADON_LOADING_BAR.options.failedColor=t;},tempColor:function(t){this.state.tColor=this.$vm.RADON_LOADING_BAR.options.color,this.$vm.RADON_LOADING_BAR.options.color=t;},tempLocation:function(t){this.state.tLocation=this.$vm.RADON_LOADING_BAR.options.location,this.$vm.RADON_LOADING_BAR.options.location=t;},tempTransition:function(t){this.state.tTransition=this.$vm.RADON_LOADING_BAR.options.transition,this.$vm.RADON_LOADING_BAR.options.transition=t;},revertColor:function(){this.$vm.RADON_LOADING_BAR.options.color=this.state.tColor,this.state.tColor="";},revertFailColor:function(){this.$vm.RADON_LOADING_BAR.options.failedColor=this.state.tFailColor,this.state.tFailColor="";},revertLocation:function(){this.$vm.RADON_LOADING_BAR.options.location=this.state.tLocation,this.state.tLocation="";},revertTransition:function(){this.$vm.RADON_LOADING_BAR.options.transition=this.state.tTransition,this.state.tTransition={};},revert:function(){this.$vm.RADON_LOADING_BAR.options.autoRevert&&(this.state.tColor&&this.revertColor(),this.state.tFailColor&&this.revertFailColor(),this.state.tLocation&&this.revertLocation(),!this.state.tTransition||void 0===this.state.tTransition.speed&&void 0===this.state.tTransition.opacity||this.revertTransition());},parseMeta:function(t){for(var o in t.func){var i=t.func[o];switch(i.call){case"color":switch(i.modifier){case"set":this.setColor(i.argument);break;case"temp":this.tempColor(i.argument);}break;case"fail":switch(i.modifier){case"set":this.setFailColor(i.argument);break;case"temp":this.tempFailColor(i.argument);}break;case"location":switch(i.modifier){case"set":this.setLocation(i.argument);break;case"temp":this.tempLocation(i.argument);}break;case"transition":switch(i.modifier){case"set":this.setTransition(i.argument);break;case"temp":this.tempTransition(i.argument);}}}}},s=function(t,o){for(var i,e,s=1;s<arguments.length;++s)for(i in e=arguments[s])Object.prototype.hasOwnProperty.call(e,i)&&(t[i]=e[i]);return t}({canSuccess:!0,show:!1,color:"#73ccec",position:"fixed",failedColor:"red",thickness:"2px",transition:{speed:"0.2s",opacity:"0.6s",termination:300},autoRevert:!0,location:"top",inverse:!1,autoFinish:!0},t),n=new o({data:{RADON_LOADING_BAR:{percent:0,options:s}}});i&&(window.VueProgressBarEventBus=n,e.init(n)),o.component("vue-progress-bar",r),o.prototype.$Progress=e;}}});
  });

  var dist = createCommonjsModule(function (module, exports) {
  !function(e,t){module.exports=t();}(window,function(){return function(n){var o={};function i(e){if(o[e])return o[e].exports;var t=o[e]={i:e,l:!1,exports:{}};return n[e].call(t.exports,t,t.exports,i),t.l=!0,t.exports}return i.m=n,i.c=o,i.d=function(e,t,n){i.o(e,t)||Object.defineProperty(e,t,{enumerable:!0,get:n});},i.r=function(e){"undefined"!=typeof Symbol&&Symbol.toStringTag&&Object.defineProperty(e,Symbol.toStringTag,{value:"Module"}),Object.defineProperty(e,"__esModule",{value:!0});},i.t=function(t,e){if(1&e&&(t=i(t)),8&e)return t;if(4&e&&"object"==typeof t&&t&&t.__esModule)return t;var n=Object.create(null);if(i.r(n),Object.defineProperty(n,"default",{enumerable:!0,value:t}),2&e&&"string"!=typeof t)for(var o in t)i.d(n,o,function(e){return t[e]}.bind(null,o));return n},i.n=function(e){var t=e&&e.__esModule?function(){return e.default}:function(){return e};return i.d(t,"a",t),t},i.o=function(e,t){return Object.prototype.hasOwnProperty.call(e,t)},i.p="/dist/",i(i.s=11)}([function(e,t,n){var o=n(6);"string"==typeof o&&(o=[[e.i,o,""]]),o.locals&&(e.exports=o.locals);(0, n(4).default)("27d83796",o,!1,{});},function(e,t,n){var o=n(8);"string"==typeof o&&(o=[[e.i,o,""]]),o.locals&&(e.exports=o.locals);(0, n(4).default)("0e783494",o,!1,{});},function(e,t,n){var o=n(10);"string"==typeof o&&(o=[[e.i,o,""]]),o.locals&&(e.exports=o.locals);(0, n(4).default)("17757f60",o,!1,{});},function(e,t){e.exports=function(n){var a=[];return a.toString=function(){return this.map(function(e){var t=function(e,t){var n=e[1]||"",o=e[3];if(!o)return n;if(t&&"function"==typeof btoa){var i=(a=o,"/*# sourceMappingURL=data:application/json;charset=utf-8;base64,"+btoa(unescape(encodeURIComponent(JSON.stringify(a))))+" */"),r=o.sources.map(function(e){return "/*# sourceURL="+o.sourceRoot+e+" */"});return [n].concat(r).concat([i]).join("\n")}var a;return [n].join("\n")}(e,n);return e[2]?"@media "+e[2]+"{"+t+"}":t}).join("")},a.i=function(e,t){"string"==typeof e&&(e=[[null,e,""]]);for(var n={},o=0;o<this.length;o++){var i=this[o][0];"number"==typeof i&&(n[i]=!0);}for(o=0;o<e.length;o++){var r=e[o];"number"==typeof r[0]&&n[r[0]]||(t&&!r[2]?r[2]=t:t&&(r[2]="("+r[2]+") and ("+t+")"),a.push(r));}},a};},function(e,t,n){function l(e,t){for(var n=[],o={},i=0;i<t.length;i++){var r=t[i],a=r[0],s={id:e+":"+i,css:r[1],media:r[2],sourceMap:r[3]};o[a]?o[a].parts.push(s):n.push(o[a]={id:a,parts:[s]});}return n}n.r(t),n.d(t,"default",function(){return p});var o="undefined"!=typeof document;if("undefined"!=typeof DEBUG&&DEBUG&&!o)throw new Error("vue-style-loader cannot be used in a non-browser environment. Use { target: 'node' } in your Webpack config to indicate a server-rendering environment.");var d={},i=o&&(document.head||document.getElementsByTagName("head")[0]),r=null,a=0,u=!1,s=function(){},c=null,h="data-vue-ssr-id",f="undefined"!=typeof navigator&&/msie [6-9]\b/.test(navigator.userAgent.toLowerCase());function p(a,e,t,n){u=t,c=n||{};var s=l(a,e);return v(s),function(e){for(var t=[],n=0;n<s.length;n++){var o=s[n];(i=d[o.id]).refs--,t.push(i);}e?v(s=l(a,e)):s=[];for(n=0;n<t.length;n++){var i;if(0===(i=t[n]).refs){for(var r=0;r<i.parts.length;r++)i.parts[r]();delete d[i.id];}}}}function v(e){for(var t=0;t<e.length;t++){var n=e[t],o=d[n.id];if(o){o.refs++;for(var i=0;i<o.parts.length;i++)o.parts[i](n.parts[i]);for(;i<n.parts.length;i++)o.parts.push(g(n.parts[i]));o.parts.length>n.parts.length&&(o.parts.length=n.parts.length);}else{var r=[];for(i=0;i<n.parts.length;i++)r.push(g(n.parts[i]));d[n.id]={id:n.id,refs:1,parts:r};}}}function m(){var e=document.createElement("style");return e.type="text/css",i.appendChild(e),e}function g(t){var n,o,e=document.querySelector("style["+h+'~="'+t.id+'"]');if(e){if(u)return s;e.parentNode.removeChild(e);}if(f){var i=a++;e=r||(r=m()),n=w.bind(null,e,i,!1),o=w.bind(null,e,i,!0);}else e=m(),n=function(e,t){var n=t.css,o=t.media,i=t.sourceMap;o&&e.setAttribute("media",o);c.ssrId&&e.setAttribute(h,t.id);i&&(n+="\n/*# sourceURL="+i.sources[0]+" */",n+="\n/*# sourceMappingURL=data:application/json;base64,"+btoa(unescape(encodeURIComponent(JSON.stringify(i))))+" */");if(e.styleSheet)e.styleSheet.cssText=n;else{for(;e.firstChild;)e.removeChild(e.firstChild);e.appendChild(document.createTextNode(n));}}.bind(null,e),o=function(){e.parentNode.removeChild(e);};return n(t),function(e){if(e){if(e.css===t.css&&e.media===t.media&&e.sourceMap===t.sourceMap)return;n(t=e);}else o();}}var b,y=(b=[],function(e,t){return b[e]=t,b.filter(Boolean).join("\n")});function w(e,t,n,o){var i=n?"":o.css;if(e.styleSheet)e.styleSheet.cssText=y(t,i);else{var r=document.createTextNode(i),a=e.childNodes;a[t]&&e.removeChild(a[t]),a.length?e.insertBefore(r,a[t]):e.appendChild(r);}}},function(e,t,n){var o=n(0);n.n(o).a;},function(e,t,n){(e.exports=n(3)(!1)).push([e.i,"\n.vue-modal-resizer {\r\n  display: block;\r\n  overflow: hidden;\r\n  position: absolute;\r\n  width: 12px;\r\n  height: 12px;\r\n  right: 0;\r\n  bottom: 0;\r\n  z-index: 9999999;\r\n  background: transparent;\r\n  cursor: se-resize;\n}\n.vue-modal-resizer::after {\r\n  display: block;\r\n  position: absolute;\r\n  content: '';\r\n  background: transparent;\r\n  left: 0;\r\n  top: 0;\r\n  width: 0;\r\n  height: 0;\r\n  border-bottom: 10px solid #ddd;\r\n  border-left: 10px solid transparent;\n}\n.vue-modal-resizer.clicked::after {\r\n  border-bottom: 10px solid #369be9;\n}\r\n",""]);},function(e,t,n){var o=n(1);n.n(o).a;},function(e,t,n){(e.exports=n(3)(!1)).push([e.i,"\n.v--modal-block-scroll {\r\n  overflow: hidden;\r\n  width: 100vw;\n}\n.v--modal-overlay {\r\n  position: fixed;\r\n  box-sizing: border-box;\r\n  left: 0;\r\n  top: 0;\r\n  width: 100%;\r\n  height: 100vh;\r\n  background: rgba(0, 0, 0, 0.2);\r\n  z-index: 999;\r\n  opacity: 1;\n}\n.v--modal-overlay.scrollable {\r\n  height: 100%;\r\n  min-height: 100vh;\r\n  overflow-y: auto;\r\n  -webkit-overflow-scrolling: touch;\n}\n.v--modal-overlay .v--modal-background-click {\r\n  width: 100%;\r\n  min-height: 100%;\r\n  height: auto;\n}\n.v--modal-overlay .v--modal-box {\r\n  position: relative;\r\n  overflow: hidden;\r\n  box-sizing: border-box;\n}\n.v--modal-overlay.scrollable .v--modal-box {\r\n  margin-bottom: 2px;\n}\n.v--modal {\r\n  background-color: white;\r\n  text-align: left;\r\n  border-radius: 3px;\r\n  box-shadow: 0 20px 60px -2px rgba(27, 33, 58, 0.4);\r\n  padding: 0;\n}\n.v--modal.v--modal-fullscreen {\r\n  width: 100vw;\r\n  height: 100vh;\r\n  margin: 0;\r\n  left: 0;\r\n  top: 0;\n}\n.v--modal-top-right {\r\n  display: block;\r\n  position: absolute;\r\n  right: 0;\r\n  top: 0;\n}\n.overlay-fade-enter-active,\r\n.overlay-fade-leave-active {\r\n  transition: all 0.2s;\n}\n.overlay-fade-enter,\r\n.overlay-fade-leave-active {\r\n  opacity: 0;\n}\n.nice-modal-fade-enter-active,\r\n.nice-modal-fade-leave-active {\r\n  transition: all 0.4s;\n}\n.nice-modal-fade-enter,\r\n.nice-modal-fade-leave-active {\r\n  opacity: 0;\r\n  transform: translateY(-20px);\n}\r\n",""]);},function(e,t,n){var o=n(2);n.n(o).a;},function(e,t,n){(e.exports=n(3)(!1)).push([e.i,"\n.vue-dialog div {\r\n  box-sizing: border-box;\n}\n.vue-dialog .dialog-flex {\r\n  width: 100%;\r\n  height: 100%;\n}\n.vue-dialog .dialog-content {\r\n  flex: 1 0 auto;\r\n  width: 100%;\r\n  padding: 15px;\r\n  font-size: 14px;\n}\n.vue-dialog .dialog-c-title {\r\n  font-weight: 600;\r\n  padding-bottom: 15px;\n}\n.vue-dialog .dialog-c-text {\n}\n.vue-dialog .vue-dialog-buttons {\r\n  display: flex;\r\n  flex: 0 1 auto;\r\n  width: 100%;\r\n  border-top: 1px solid #eee;\n}\n.vue-dialog .vue-dialog-buttons-none {\r\n  width: 100%;\r\n  padding-bottom: 15px;\n}\n.vue-dialog-button {\r\n  font-size: 12px !important;\r\n  background: transparent;\r\n  padding: 0;\r\n  margin: 0;\r\n  border: 0;\r\n  cursor: pointer;\r\n  box-sizing: border-box;\r\n  line-height: 40px;\r\n  height: 40px;\r\n  color: inherit;\r\n  font: inherit;\r\n  outline: none;\n}\n.vue-dialog-button:hover {\r\n  background: rgba(0, 0, 0, 0.01);\n}\n.vue-dialog-button:active {\r\n  background: rgba(0, 0, 0, 0.025);\n}\n.vue-dialog-button:not(:first-of-type) {\r\n  border-left: 1px solid #eee;\n}\r\n",""]);},function(e,t,n){n.r(t);var o=function(){var t=this,e=t.$createElement,n=t._self._c||e;return n("transition",{attrs:{name:t.overlayTransition}},[t.visibility.overlay?n("div",{ref:"overlay",class:t.overlayClass,attrs:{"aria-expanded":t.visibility.overlay.toString(),"data-modal":t.name}},[n("div",{staticClass:"v--modal-background-click",on:{mousedown:function(e){return e.target!==e.currentTarget?null:t.handleBackgroundClick(e)},touchstart:function(e){return e.target!==e.currentTarget?null:t.handleBackgroundClick(e)}}},[n("div",{staticClass:"v--modal-top-right"},[t._t("top-right")],2),t._v(" "),n("transition",{attrs:{name:t.transition},on:{"before-enter":t.beforeTransitionEnter,"after-enter":t.afterTransitionEnter,"after-leave":t.afterTransitionLeave}},[t.visibility.modal?n("div",{ref:"modal",class:t.modalClass,style:t.modalStyle},[t._t("default"),t._v(" "),t.resizable&&!t.isAutoHeight?n("resizer",{attrs:{"min-width":t.minWidth,"min-height":t.minHeight},on:{resize:t.handleModalResize}}):t._e()],2):t._e()])],1)]):t._e()])},i=function(){var e=this.$createElement;return (this._self._c||e)("div",{class:this.className})};i._withStripped=o._withStripped=!0;var s=function(){var e=0<arguments.length&&void 0!==arguments[0]?arguments[0]:0;return function(){return (e++).toString()}}(),u=function(e,t,n){return n<e?e:t<n?t:n},r=function(){var e=0<arguments.length&&void 0!==arguments[0]?arguments[0]:{};return function(i){for(var e=1;e<arguments.length;e++){var r=null!=arguments[e]?arguments[e]:{},t=Object.keys(r);"function"==typeof Object.getOwnPropertySymbols&&(t=t.concat(Object.getOwnPropertySymbols(r).filter(function(e){return Object.getOwnPropertyDescriptor(r,e).enumerable}))),t.forEach(function(e){var t,n,o;t=i,o=r[n=e],n in t?Object.defineProperty(t,n,{value:o,enumerable:!0,configurable:!0,writable:!0}):t[n]=o;});}return i}({id:s(),timestamp:Date.now(),canceled:!1},e)},a={name:"VueJsModalResizer",props:{minHeight:{type:Number,default:0},minWidth:{type:Number,default:0}},data:function(){return {clicked:!1,size:{}}},mounted:function(){this.$el.addEventListener("mousedown",this.start,!1);},computed:{className:function(){return {"vue-modal-resizer":!0,clicked:this.clicked}}},methods:{start:function(e){this.clicked=!0,window.addEventListener("mousemove",this.mousemove,!1),window.addEventListener("mouseup",this.stop,!1),e.stopPropagation(),e.preventDefault();},stop:function(){this.clicked=!1,window.removeEventListener("mousemove",this.mousemove,!1),window.removeEventListener("mouseup",this.stop,!1),this.$emit("resize-stop",{element:this.$el.parentElement,size:this.size});},mousemove:function(e){this.resize(e);},resize:function(e){var t=this.$el.parentElement;if(t){var n=e.clientX-t.offsetLeft,o=e.clientY-t.offsetTop;n=u(this.minWidth,window.innerWidth,n),o=u(this.minHeight,window.innerHeight,o),this.size={width:n,height:o},t.style.width=n+"px",t.style.height=o+"px",this.$emit("resize",{element:t,size:this.size});}}}};n(5);function l(e,t,n,o,i,r,a,s){var l,d="function"==typeof e?e.options:e;if(t&&(d.render=t,d.staticRenderFns=n,d._compiled=!0),o&&(d.functional=!0),r&&(d._scopeId="data-v-"+r),a?(l=function(e){(e=e||this.$vnode&&this.$vnode.ssrContext||this.parent&&this.parent.$vnode&&this.parent.$vnode.ssrContext)||"undefined"==typeof __VUE_SSR_CONTEXT__||(e=__VUE_SSR_CONTEXT__),i&&i.call(this,e),e&&e._registeredComponents&&e._registeredComponents.add(a);},d._ssrRegister=l):i&&(l=s?function(){i.call(this,this.$root.$options.shadowRoot);}:i),l)if(d.functional){d._injectStyles=l;var u=d.render;d.render=function(e,t){return l.call(t),u(e,t)};}else{var c=d.beforeCreate;d.beforeCreate=c?[].concat(c,l):[l];}return {exports:e,options:d}}var d=l(a,i,[],!1,null,null,null);d.options.__file="src/Resizer.vue";var c=d.exports;function h(e){return (h="function"==typeof Symbol&&"symbol"==typeof Symbol.iterator?function(e){return typeof e}:function(e){return e&&"function"==typeof Symbol&&e.constructor===Symbol&&e!==Symbol.prototype?"symbol":typeof e})(e)}var f="[-+]?[0-9]*.?[0-9]+",p=[{name:"px",regexp:new RegExp("^".concat(f,"px$"))},{name:"%",regexp:new RegExp("^".concat(f,"%$"))},{name:"px",regexp:new RegExp("^".concat(f,"$"))}],v=function(e){switch(h(e)){case"number":return {type:"px",value:e};case"string":return function(e){if("auto"===e)return {type:e,value:0};for(var t=0;t<p.length;t++){var n=p[t];if(n.regexp.test(e))return {type:n.name,value:parseFloat(e)}}return {type:"",value:e}}(e);default:return {type:"",value:e}}},m=function(e){if("string"!=typeof e)return 0<=e;var t=v(e);return ("%"===t.type||"px"===t.type)&&0<t.value};var g={name:"VueJsModal",props:{name:{required:!0,type:String},delay:{type:Number,default:0},resizable:{type:Boolean,default:!1},adaptive:{type:Boolean,default:!1},draggable:{type:[Boolean,String],default:!1},scrollable:{type:Boolean,default:!1},reset:{type:Boolean,default:!1},overlayTransition:{type:String,default:"overlay-fade"},transition:{type:String},clickToClose:{type:Boolean,default:!0},classes:{type:[String,Array],default:"v--modal"},minWidth:{type:Number,default:0,validator:function(e){return 0<=e}},minHeight:{type:Number,default:0,validator:function(e){return 0<=e}},maxWidth:{type:Number,default:1/0},maxHeight:{type:Number,default:1/0},width:{type:[Number,String],default:600,validator:m},height:{type:[Number,String],default:300,validator:function(e){return "auto"===e||m(e)}},pivotX:{type:Number,default:.5,validator:function(e){return 0<=e&&e<=1}},pivotY:{type:Number,default:.5,validator:function(e){return 0<=e&&e<=1}}},components:{Resizer:c},data:function(){return {visible:!1,visibility:{modal:!1,overlay:!1},shift:{left:0,top:0},modal:{width:0,widthType:"px",height:0,heightType:"px",renderedHeight:0},window:{width:0,height:0},mutationObserver:null}},created:function(){this.setInitialSize();},beforeMount:function(){var t=this;if(z.event.$on("toggle",this.handleToggleEvent),window.addEventListener("resize",this.handleWindowResize),this.handleWindowResize(),this.scrollable&&!this.isAutoHeight&&console.warn('Modal "'.concat(this.name,'" has scrollable flag set to true ')+'but height is not "auto" ('.concat(this.height,")")),this.isAutoHeight){var e=function(){if("undefined"!=typeof window)for(var e=["","WebKit","Moz","O","Ms"],t=0;t<e.length;t++){var n=e[t]+"MutationObserver";if(n in window)return window[n]}return !1}();e&&(this.mutationObserver=new e(function(e){t.updateRenderedHeight();}));}this.clickToClose&&window.addEventListener("keyup",this.handleEscapeKeyUp);},beforeDestroy:function(){z.event.$off("toggle",this.handleToggleEvent),window.removeEventListener("resize",this.handleWindowResize),this.clickToClose&&window.removeEventListener("keyup",this.handleEscapeKeyUp),this.scrollable&&document.body.classList.remove("v--modal-block-scroll");},computed:{isAutoHeight:function(){return "auto"===this.modal.heightType},position:function(){var e=this.window,t=this.shift,n=this.pivotX,o=this.pivotY,i=this.trueModalWidth,r=this.trueModalHeight,a=e.width-i,s=e.height-r,l=t.left+n*a,d=t.top+o*s;return {left:parseInt(u(0,a,l)),top:parseInt(u(0,s,d))}},trueModalWidth:function(){var e=this.window,t=this.modal,n=this.adaptive,o=this.minWidth,i=this.maxWidth,r="%"===t.widthType?e.width/100*t.width:t.width,a=Math.min(e.width,i);return n?u(o,a,r):r},trueModalHeight:function(){var e=this.window,t=this.modal,n=this.isAutoHeight,o=this.adaptive,i=this.maxHeight,r="%"===t.heightType?e.height/100*t.height:t.height;if(n)return this.modal.renderedHeight;var a=Math.min(e.height,i);return o?u(this.minHeight,a,r):r},overlayClass:function(){return {"v--modal-overlay":!0,scrollable:this.scrollable&&this.isAutoHeight}},modalClass:function(){return ["v--modal-box",this.classes]},modalStyle:function(){return {top:this.position.top+"px",left:this.position.left+"px",width:this.trueModalWidth+"px",height:this.isAutoHeight?"auto":this.trueModalHeight+"px"}}},watch:{visible:function(e){var t=this;e?(this.visibility.overlay=!0,setTimeout(function(){t.visibility.modal=!0,t.$nextTick(function(){t.addDraggableListeners(),t.callAfterEvent(!0);});},this.delay)):(this.visibility.modal=!1,setTimeout(function(){t.visibility.overlay=!1,t.$nextTick(function(){t.removeDraggableListeners(),t.callAfterEvent(!1);});},this.delay));}},methods:{handleToggleEvent:function(e,t,n){if(this.name===e){var o=void 0===t?!this.visible:t;this.toggle(o,n);}},setInitialSize:function(){var e=this.modal,t=v(this.width),n=v(this.height);e.width=t.value,e.widthType=t.type,e.height=n.value,e.heightType=n.type;},handleEscapeKeyUp:function(e){27===e.which&&this.visible&&this.$modal.hide(this.name);},handleWindowResize:function(){this.window.width=window.innerWidth,this.window.height=window.innerHeight,this.ensureShiftInWindowBounds();},createModalEvent:function(){var e=0<arguments.length&&void 0!==arguments[0]?arguments[0]:{};return r(function(i){for(var e=1;e<arguments.length;e++){var r=null!=arguments[e]?arguments[e]:{},t=Object.keys(r);"function"==typeof Object.getOwnPropertySymbols&&(t=t.concat(Object.getOwnPropertySymbols(r).filter(function(e){return Object.getOwnPropertyDescriptor(r,e).enumerable}))),t.forEach(function(e){var t,n,o;t=i,o=r[n=e],n in t?Object.defineProperty(t,n,{value:o,enumerable:!0,configurable:!0,writable:!0}):t[n]=o;});}return i}({name:this.name,ref:this.$refs.modal},e))},handleModalResize:function(e){this.modal.widthType="px",this.modal.width=e.size.width,this.modal.heightType="px",this.modal.height=e.size.height;var t=this.modal.size;this.$emit("resize",this.createModalEvent({size:t}));},toggle:function(e,t){var n=this.reset,o=this.scrollable,i=this.visible;if(i!==e){var r=i?"before-close":"before-open";"before-open"===r?("undefined"!=typeof document&&document.activeElement&&"BODY"!==document.activeElement.tagName&&document.activeElement.blur&&document.activeElement.blur(),n&&(this.setInitialSize(),this.shift.left=0,this.shift.top=0),o&&document.body.classList.add("v--modal-block-scroll")):o&&document.body.classList.remove("v--modal-block-scroll");var a=!1,s=this.createModalEvent({stop:function(){a=!0;},state:e,params:t});this.$emit(r,s),a||(this.visible=e);}},getDraggableElement:function(){var e="string"!=typeof this.draggable?".v--modal-box":this.draggable;return e?this.$refs.overlay.querySelector(e):null},handleBackgroundClick:function(){this.clickToClose&&this.toggle(!1);},callAfterEvent:function(e){e?this.connectObserver():this.disconnectObserver();var t=e?"opened":"closed",n=this.createModalEvent({state:e});this.$emit(t,n);},addDraggableListeners:function(){var r=this;if(this.draggable){var e=this.getDraggableElement();if(e){var a=0,s=0,l=0,d=0,u=function(e){return e.touches&&0<e.touches.length?e.touches[0]:e},t=function(e){var t=e.target;if(!t||"INPUT"!==t.nodeName){var n=u(e),o=n.clientX,i=n.clientY;document.addEventListener("mousemove",c),document.addEventListener("touchmove",c),document.addEventListener("mouseup",h),document.addEventListener("touchend",h),a=o,s=i,l=r.shift.left,d=r.shift.top;}},c=function(e){var t=u(e),n=t.clientX,o=t.clientY;r.shift.left=l+n-a,r.shift.top=d+o-s,e.preventDefault();},h=function e(t){r.ensureShiftInWindowBounds(),document.removeEventListener("mousemove",c),document.removeEventListener("touchmove",c),document.removeEventListener("mouseup",e),document.removeEventListener("touchend",e),t.preventDefault();};e.addEventListener("mousedown",t),e.addEventListener("touchstart",t);}}},removeDraggableListeners:function(){},updateRenderedHeight:function(){this.$refs.modal&&(this.modal.renderedHeight=this.$refs.modal.getBoundingClientRect().height);},connectObserver:function(){this.mutationObserver&&this.mutationObserver.observe(this.$refs.overlay,{childList:!0,attributes:!0,subtree:!0});},disconnectObserver:function(){this.mutationObserver&&this.mutationObserver.disconnect();},beforeTransitionEnter:function(){this.connectObserver();},afterTransitionEnter:function(){},afterTransitionLeave:function(){},ensureShiftInWindowBounds:function(){var e=this.window,t=this.shift,n=this.pivotX,o=this.pivotY,i=this.trueModalWidth,r=this.trueModalHeight,a=e.width-i,s=e.height-r,l=t.left+n*a,d=t.top+o*s;this.shift.left-=l-u(0,a,l),this.shift.top-=d-u(0,s,d);}}},b=(n(7),l(g,o,[],!1,null,null,null));b.options.__file="src/Modal.vue";var y=b.exports,w=function(){var n=this,e=n.$createElement,o=n._self._c||e;return o("modal",{attrs:{name:"dialog",height:"auto",classes:["v--modal","vue-dialog",this.params.class],width:n.width,"pivot-y":.3,adaptive:!0,clickToClose:n.clickToClose,transition:n.transition},on:{"before-open":n.beforeOpened,"before-close":n.beforeClosed,opened:function(e){n.$emit("opened",e);},closed:function(e){n.$emit("closed",e);}}},[o("div",{staticClass:"dialog-content"},[n.params.title?o("div",{staticClass:"dialog-c-title",domProps:{innerHTML:n._s(n.params.title||"")}}):n._e(),n._v(" "),n.params.component?o(n.params.component,n._b({tag:"component"},"component",n.params.props,!1)):o("div",{staticClass:"dialog-c-text",domProps:{innerHTML:n._s(n.params.text||"")}})],1),n._v(" "),n.buttons?o("div",{staticClass:"vue-dialog-buttons"},n._l(n.buttons,function(e,t){return o("button",{key:t,class:e.class||"vue-dialog-button",style:n.buttonStyle,attrs:{type:"button"},domProps:{innerHTML:n._s(e.title)},on:{click:function(e){e.stopPropagation(),n.click(t,e);}}},[n._v("\n      "+n._s(e.title)+"\n    ")])})):o("div",{staticClass:"vue-dialog-buttons-none"})])};w._withStripped=!0;var x={name:"VueJsDialog",props:{width:{type:[Number,String],default:400},clickToClose:{type:Boolean,default:!0},transition:{type:String,default:"fade"}},data:function(){return {params:{},defaultButtons:[{title:"CLOSE"}]}},computed:{buttons:function(){return this.params.buttons||this.defaultButtons},buttonStyle:function(){return {flex:"1 1 ".concat(100/this.buttons.length,"%")}}},methods:{beforeOpened:function(e){window.addEventListener("keyup",this.onKeyUp),this.params=e.params||{},this.$emit("before-opened",e);},beforeClosed:function(e){window.removeEventListener("keyup",this.onKeyUp),this.params={},this.$emit("before-closed",e);},click:function(e,t){var n=2<arguments.length&&void 0!==arguments[2]?arguments[2]:"click",o=this.buttons[e];o&&"function"==typeof o.handler?o.handler(e,t,{source:n}):this.$modal.hide("dialog");},onKeyUp:function(e){if(13===e.which&&0<this.buttons.length){var t=1===this.buttons.length?0:this.buttons.findIndex(function(e){return e.default});-1!==t&&this.click(t,e,"keypress");}}}},_=(n(9),l(x,w,[],!1,null,null,null));_.options.__file="src/Dialog.vue";var E=_.exports,S=function(){var n=this,e=n.$createElement,o=n._self._c||e;return o("div",{attrs:{id:"modals-container"}},n._l(n.modals,function(t){return o("modal",n._g(n._b({key:t.id,on:{closed:function(e){n.remove(t.id);}}},"modal",t.modalAttrs,!1),t.modalListeners),[o(t.component,n._g(n._b({tag:"component",on:{close:function(e){n.$modal.hide(t.modalAttrs.name);}}},"component",t.componentAttrs,!1),n.$listeners))],1)}))};S._withStripped=!0;var O=l({data:function(){return {modals:[]}},created:function(){this.$root._dynamicContainer=this;},methods:{add:function(e){var t=this,n=1<arguments.length&&void 0!==arguments[1]?arguments[1]:{},o=2<arguments.length&&void 0!==arguments[2]?arguments[2]:{},i=3<arguments.length&&void 0!==arguments[3]?arguments[3]:{},r=s(),a=o.name||"_dynamic_modal_"+r;this.modals.push({id:r,modalAttrs:function(i){for(var e=1;e<arguments.length;e++){var r=null!=arguments[e]?arguments[e]:{},t=Object.keys(r);"function"==typeof Object.getOwnPropertySymbols&&(t=t.concat(Object.getOwnPropertySymbols(r).filter(function(e){return Object.getOwnPropertyDescriptor(r,e).enumerable}))),t.forEach(function(e){var t,n,o;t=i,o=r[n=e],n in t?Object.defineProperty(t,n,{value:o,enumerable:!0,configurable:!0,writable:!0}):t[n]=o;});}return i}({},o,{name:a}),modalListeners:i,component:e,componentAttrs:n}),this.$nextTick(function(){t.$modal.show(a);});},remove:function(t){var e=this.modals.findIndex(function(e){return e.id===t});-1!==e&&this.modals.splice(e,1);}}},S,[],!1,null,null,null);O.options.__file="src/ModalsContainer.vue";var k=O.exports;function C(e){return (C="function"==typeof Symbol&&"symbol"==typeof Symbol.iterator?function(e){return typeof e}:function(e){return e&&"function"==typeof Symbol&&e.constructor===Symbol&&e!==Symbol.prototype?"symbol":typeof e})(e)}n.d(t,"getModalsContainer",function(){return T});var T=function(e,t,n){if(!n._dynamicContainer&&t.injectModalsContainer){var o=(i=document.createElement("div"),document.body.appendChild(i),i);new e({parent:n,render:function(e){return e(k)}}).$mount(o);}var i;return n._dynamicContainer},$={install:function(a){var s=1<arguments.length&&void 0!==arguments[1]?arguments[1]:{};if(!this.installed){this.installed=!0,this.event=new a,this.rootInstance=null;var e=s.componentName||"Modal",l=s.dynamicDefaults||{},i=function(e,t,n,o){var i=n&&n.root?n.root:$.rootInstance,r=T(a,s,i);r?r.add(e,function(i){for(var e=1;e<arguments.length;e++){var r=null!=arguments[e]?arguments[e]:{},t=Object.keys(r);"function"==typeof Object.getOwnPropertySymbols&&(t=t.concat(Object.getOwnPropertySymbols(r).filter(function(e){return Object.getOwnPropertyDescriptor(r,e).enumerable}))),t.forEach(function(e){var t,n,o;t=i,o=r[n=e],n in t?Object.defineProperty(t,n,{value:o,enumerable:!0,configurable:!0,writable:!0}):t[n]=o;});}return i}({},l,t),n,o):console.warn("[vue-js-modal] In order to render dynamic modals, a <modals-container> component must be present on the page.");};a.prototype.$modal={show:function(e){for(var t=arguments.length,n=new Array(1<t?t-1:0),o=1;o<t;o++)n[o-1]=arguments[o];switch(C(e)){case"string":return function(e,t){$.event.$emit("toggle",e,!0,t);}.apply(void 0,[e].concat(n));case"object":return s.dynamic?i.apply(void 0,[e].concat(n)):console.warn("[vue-js-modal] $modal() received object as a first argument, but dynamic modals are switched off. https://github.com/euvl/vue-js-modal/#dynamic-modals")}},hide:function(e,t){$.event.$emit("toggle",e,!1,t);},toggle:function(e,t){$.event.$emit("toggle",e,void 0,t);}},a.component(e,y),s.dialog&&a.component("VDialog",E),s.dynamic&&(a.component("ModalsContainer",k),a.mixin({beforeMount:function(){null===$.rootInstance&&($.rootInstance=this.$root);}}));}}},z=t.default=$;}])});
  });

  var VModal = unwrapExports(dist);

  var Vue$1 = Vue;
  Vue$1 = 'default' in Vue$1 ? Vue$1['default'] : Vue$1;

  var version = '2.2.2';

  var compatible = (/^2\./).test(Vue$1.version);
  if (!compatible) {
    Vue$1.util.warn('VueClickaway ' + version + ' only supports Vue 2.x, and does not support Vue ' + Vue$1.version);
  }



  // @SECTION: implementation

  var HANDLER = '_vue_clickaway_handler';

  function bind$2(el, binding, vnode) {
    unbind(el);

    var vm = vnode.context;

    var callback = binding.value;
    if (typeof callback !== 'function') {
      {
        Vue$1.util.warn(
          'v-' + binding.name + '="' +
          binding.expression + '" expects a function value, ' +
          'got ' + callback
        );
      }
      return;
    }

    // @NOTE: Vue binds directives in microtasks, while UI events are dispatched
    //        in macrotasks. This causes the listener to be set up before
    //        the "origin" click event (the event that lead to the binding of
    //        the directive) arrives at the document root. To work around that,
    //        we ignore events until the end of the "initial" macrotask.
    // @REFERENCE: https://jakearchibald.com/2015/tasks-microtasks-queues-and-schedules/
    // @REFERENCE: https://github.com/simplesmiler/vue-clickaway/issues/8
    var initialMacrotaskEnded = false;
    setTimeout(function() {
      initialMacrotaskEnded = true;
    }, 0);

    el[HANDLER] = function(ev) {
      // @NOTE: this test used to be just `el.containts`, but working with path is better,
      //        because it tests whether the element was there at the time of
      //        the click, not whether it is there now, that the event has arrived
      //        to the top.
      // @NOTE: `.path` is non-standard, the standard way is `.composedPath()`
      var path = ev.path || (ev.composedPath ? ev.composedPath() : undefined);
      if (initialMacrotaskEnded && (path ? path.indexOf(el) < 0 : !el.contains(ev.target))) {
        return callback.call(vm, ev);
      }
    };

    document.documentElement.addEventListener('click', el[HANDLER], false);
  }

  function unbind(el) {
    document.documentElement.removeEventListener('click', el[HANDLER], false);
    delete el[HANDLER];
  }

  var directive$1 = {
    bind: bind$2,
    update: function(el, binding) {
      if (binding.value === binding.oldValue) return;
      bind$2(el, binding);
    },
    unbind: unbind,
  };
  var directive_1 = directive$1;

  var vueDialogDrag_umd = createCommonjsModule(function (module, exports) {
  (function webpackUniversalModuleDefinition(root, factory) {
  	module.exports = factory();
  })(typeof self !== 'undefined' ? self : commonjsGlobal, function() {
  return /******/ (function(modules) { // webpackBootstrap
  /******/ 	// The module cache
  /******/ 	var installedModules = {};
  /******/
  /******/ 	// The require function
  /******/ 	function __webpack_require__(moduleId) {
  /******/
  /******/ 		// Check if module is in cache
  /******/ 		if(installedModules[moduleId]) {
  /******/ 			return installedModules[moduleId].exports;
  /******/ 		}
  /******/ 		// Create a new module (and put it into the cache)
  /******/ 		var module = installedModules[moduleId] = {
  /******/ 			i: moduleId,
  /******/ 			l: false,
  /******/ 			exports: {}
  /******/ 		};
  /******/
  /******/ 		// Execute the module function
  /******/ 		modules[moduleId].call(module.exports, module, module.exports, __webpack_require__);
  /******/
  /******/ 		// Flag the module as loaded
  /******/ 		module.l = true;
  /******/
  /******/ 		// Return the exports of the module
  /******/ 		return module.exports;
  /******/ 	}
  /******/
  /******/
  /******/ 	// expose the modules object (__webpack_modules__)
  /******/ 	__webpack_require__.m = modules;
  /******/
  /******/ 	// expose the module cache
  /******/ 	__webpack_require__.c = installedModules;
  /******/
  /******/ 	// define getter function for harmony exports
  /******/ 	__webpack_require__.d = function(exports, name, getter) {
  /******/ 		if(!__webpack_require__.o(exports, name)) {
  /******/ 			Object.defineProperty(exports, name, { enumerable: true, get: getter });
  /******/ 		}
  /******/ 	};
  /******/
  /******/ 	// define __esModule on exports
  /******/ 	__webpack_require__.r = function(exports) {
  /******/ 		if(typeof Symbol !== 'undefined' && Symbol.toStringTag) {
  /******/ 			Object.defineProperty(exports, Symbol.toStringTag, { value: 'Module' });
  /******/ 		}
  /******/ 		Object.defineProperty(exports, '__esModule', { value: true });
  /******/ 	};
  /******/
  /******/ 	// create a fake namespace object
  /******/ 	// mode & 1: value is a module id, require it
  /******/ 	// mode & 2: merge all properties of value into the ns
  /******/ 	// mode & 4: return value when already ns object
  /******/ 	// mode & 8|1: behave like require
  /******/ 	__webpack_require__.t = function(value, mode) {
  /******/ 		if(mode & 1) value = __webpack_require__(value);
  /******/ 		if(mode & 8) return value;
  /******/ 		if((mode & 4) && typeof value === 'object' && value && value.__esModule) return value;
  /******/ 		var ns = Object.create(null);
  /******/ 		__webpack_require__.r(ns);
  /******/ 		Object.defineProperty(ns, 'default', { enumerable: true, value: value });
  /******/ 		if(mode & 2 && typeof value != 'string') for(var key in value) __webpack_require__.d(ns, key, function(key) { return value[key]; }.bind(null, key));
  /******/ 		return ns;
  /******/ 	};
  /******/
  /******/ 	// getDefaultExport function for compatibility with non-harmony modules
  /******/ 	__webpack_require__.n = function(module) {
  /******/ 		var getter = module && module.__esModule ?
  /******/ 			function getDefault() { return module['default']; } :
  /******/ 			function getModuleExports() { return module; };
  /******/ 		__webpack_require__.d(getter, 'a', getter);
  /******/ 		return getter;
  /******/ 	};
  /******/
  /******/ 	// Object.prototype.hasOwnProperty.call
  /******/ 	__webpack_require__.o = function(object, property) { return Object.prototype.hasOwnProperty.call(object, property); };
  /******/
  /******/ 	// __webpack_public_path__
  /******/ 	__webpack_require__.p = "";
  /******/
  /******/
  /******/ 	// Load entry module and return exports
  /******/ 	return __webpack_require__(__webpack_require__.s = "+xUi");
  /******/ })
  /************************************************************************/
  /******/ ({

  /***/ "+rLv":
  /***/ (function(module, exports, __webpack_require__) {

  var document = __webpack_require__("dyZX").document;
  module.exports = document && document.documentElement;


  /***/ }),

  /***/ "+xUi":
  /***/ (function(module, __webpack_exports__, __webpack_require__) {
  __webpack_require__.r(__webpack_exports__);

  // EXTERNAL MODULE: ./node_modules/@vue/cli-service/lib/commands/build/setPublicPath.js
  var setPublicPath = __webpack_require__("HrLf");

  // CONCATENATED MODULE: ./node_modules/cache-loader/dist/cjs.js?{"cacheDirectory":"/var/share/vue-dialog-drag/node_modules/.cache/vue-loader","cacheIdentifier":"847cbeee-vue-loader-template"}!./node_modules/vue-loader/lib/loaders/templateLoader.js??vue-loader-options!./node_modules/pug-plain-loader!./node_modules/cache-loader/dist/cjs.js??ref--0-0!./node_modules/vue-loader/lib??vue-loader-options!./src/components/vue-dialog-drag.vue?vue&type=template&id=1c049c8d&lang=pug&
  var render = function () {var _vm=this;var _h=_vm.$createElement;var _c=_vm._self._c||_h;return _c('div',{staticClass:"dialog-drag",class:(!_vm.drag) ? "fixed":"",style:(_vm.dialogStyle),attrs:{"id":_vm.id,"draggable":_vm.drag},on:{"mousedown":_vm.mouseDown,"touchstart":function($event){$event.preventDefault();return _vm.touchStart($event)},"&touchmove":function($event){return _vm.touchMove($event)},"touchend":function($event){$event.stopPropagation();return _vm.touchEnd($event)}}},[_c('div',{staticClass:"dialog-header",on:{"dragstart":function($event){$event.stopPropagation();}}},[_c('div',{staticClass:"title"},[_vm._t("title",[(_vm.title)?_c('span',[_vm._v(_vm._s(_vm.title))]):_c('span',[_vm._v(" ")])])],2),_c('div',{staticClass:"buttons"},[(_vm.buttonPin)?_c('button',{staticClass:"pin",on:{"click":_vm.setDrag,"touchstart":_vm.setDrag}},[(_vm.drag)?_vm._t("button-pin"):_vm._e(),(!_vm.drag)?_vm._t("button-pinned",[(!_vm.drag)?_vm._t("button-pin"):_vm._e()]):_vm._e()],2):_vm._e(),(_vm.buttonClose)?_c('button',{staticClass:"close",on:{"click":function($event){$event.stopPropagation();return _vm.close($event)},"&touchstart":function($event){return _vm.close($event)}}},[_vm._t("button-close")],2):_vm._e()])]),_c('div',{staticClass:"dialog-body",on:{"dragstart":function($event){$event.stopPropagation();}}},[_vm._t("default",[_c('div',{staticClass:"blank-body"})])],2)])};
  var staticRenderFns = [];


  // CONCATENATED MODULE: ./src/components/vue-dialog-drag.vue?vue&type=template&id=1c049c8d&lang=pug&

  // EXTERNAL MODULE: ./node_modules/core-js/modules/es7.symbol.async-iterator.js
  var es7_symbol_async_iterator = __webpack_require__("rE2o");

  // EXTERNAL MODULE: ./node_modules/core-js/modules/es6.symbol.js
  var es6_symbol = __webpack_require__("ioFf");

  // EXTERNAL MODULE: ./node_modules/core-js/modules/web.dom.iterable.js
  var web_dom_iterable = __webpack_require__("rGqo");

  // CONCATENATED MODULE: ./node_modules/cache-loader/dist/cjs.js??ref--12-0!./node_modules/babel-loader/lib!./node_modules/vue-loader/lib??vue-loader-options!./src/components/vue-dialog-drag.vue?vue&type=script&lang=js&



  //
  //
  //
  //
  //
  //
  //
  //
  //
  //
  //
  //
  //
  //
  //
  //
  //
  //
  //
  //
  //
  //
  //
  //
  //
  //
  //
  //
  //
  //
  //
  //
  //
  //
  //
  //
  /* harmony default export */ var vue_dialog_dragvue_type_script_lang_js_ = ({
    name: 'dialog-drag',
    props: ['id', 'title', 'options', 'eventCb'],
    data: function data() {
      return {
        width: 0,
        height: 0,
        zIndex: 0,
        offset: {
          x: 0,
          y: 0
        },
        left: 0,
        top: 0,
        buttonClose: true,
        buttonPin: true,
        dragEnabled: true,
        drag: true,
        touch: null,
        overEvent: null,
        centered: false,
        dropEnabled: true,
        dragCursor: 'default',
        dragging: false,
        clickButton: false,
        pX: 0,
        pY: 0,
        availableOptions: ['left', 'top', 'width', 'height', 'buttonPin', 'buttonClose', 'centered', 'dropEnabled', 'dragCursor', 'zIndex']
      };
    },
    created: function created() {
      this.setOptions(this.options);
    },
    mounted: function mounted() {
      if (this.dropEnabled) {
        this.$el.addEventListener('dragstart', this.dragStart);
        this.$el.addEventListener('dragend', this.dragEnd);
        window.addEventListener('dragover', this.dragOver);
      } else {
        document.addEventListener('mousemove', this.mouseMove, {
          passive: true
        });
        document.addEventListener('mouseup', this.mouseUp);
      }

      if (this.centered) {
        var vm = this;
        this.$nextTick(function () {
          vm.center();
          vm.emit('load');
        });
      } else {
        this.emit('load');
      }
    },
    beforeDestroy: function beforeDestroy() {
      if (this.dropEnabled) {
        window.removeEventListener('dragover', this.dragOver);
      } else {
        document.removeEventListener('mousemove', this.mouseMove);
        document.removeEventListener('mouseup', this.mouseUp);
      }
    },
    watch: {
      options: function options(newValue) {
        this.setOptions(newValue);
        if (newValue.centered) this.center();
      }
    },
    computed: {
      dialogStyle: function dialogStyle() {
        var style = {
          left: this.left + 'px',
          top: this.top + 'px'
        };
        if (this.width) style.width = this.width + 'px';
        if (this.height) style.height = this.height + 'px';
        if (this.zIndex) style.zIndex = this.zIndex;

        if (this.drag) {
          style['user-select'] = 'none';
          style.cursor = this.dragCursor;
        }

        return style;
      }
    },
    methods: {
      mouseOut: function mouseOut(event) {
        if (!this.dragEnabled && this.dragging) {
          this.move(event);
        }
      },
      dragOver: function dragOver(event) {
        if (this.dropEnabled) {
          this.overEvent = event;
          this.emit('move');
        }
      },
      mouseOver: function mouseOver(event) {
        setTimeout(this.mouseMove(event), 50);
      },
      close: function close() {
        this.clickButton = 'close';
        this.emit('close');
      },
      setDrag: function setDrag() {
        if (this.dragEnabled) {
          this.drag = !this.drag;
          this.emit('pin');
        }
      },
      dragStart: function dragStart(event) {
        event.stopPropagation();

        if (this.drag && this.dragEnabled && this.dropEnabled) {
          event.dataTransfer.setData('text', event.target.id);
          this.startMove(event);
        }
      },
      dragEnd: function dragEnd(event) {
        event.preventDefault();

        if (this.dropEnabled) {
          this.move(event);
          this.emit('drag-end');
        }
      },
      mouseDown: function mouseDown(event) {
        if (!this.dragging) this.focus();

        if (!this.dropEnabled) {
          if (this.drag) event.preventDefault();
          this.startMove(event);
        }
      },
      mouseMove: function mouseMove(event) {
        // event.preventDefault()
        if (!this.dropEnabled && this.dragging && this.drag) {
          // event.stopPropagation()
          setTimeout(this.move(event), 50);
        }
      },
      mouseUp: function mouseUp(event) {
        event.preventDefault();

        if (!this.dropEnabled) {
          this.stopMove();
          this.emit('dragEnd');
        }
      },
      touchStart: function touchStart(event) {
        this.emit('focus');
        this.startMove(event.targetTouches[0]);
      },
      touchMove: function touchMove(event) {
        this.move(event.targetTouches[0]);
      },
      touchEnd: function touchEnd(event) {
        this.emit('dragEnd');
        this.stopMove();
      },
      stopMove: function stopMove() {
        this.dragging = false;
        this.pX = 0;
        this.pY = 0;
      },
      emit: function emit(eventName, data) {
        data = data || {
          id: this.id,
          left: this.left,
          top: this.top,
          x: this.left,
          y: this.top,
          z: this.zIndex,
          pinned: !this.drag,
          width: this.$el.clientWidth,
          height: this.$el.clientHeight
        };

        if (this.eventCb) {
          var ef = this.eventCb;

          if (ef && typeof ef === 'function') {
            data = ef(data);
          }
        }

        this.$emit(eventName, data);
      },
      move: function move(event) {
        if (this.drag && this.dragEnabled) {
          if (event.clientX === 0) event = this.overEvent; // for firefox

          if (event && event.clientX && event.clientY) {
            var x = event.clientX;
            var y = event.clientY;
            this.left = x + this.offset.x;
            this.top = y + this.offset.y;
            this.dragging++;
            this.emit('move');
          }
        }
      },
      clearSelection: function clearSelection() {
        if (document.selection) {
          document.selection.empty();
        } else if (window.getSelection) {
          window.getSelection().removeAllRanges();
        }
      },
      startMove: function startMove(event) {
        var x = this.left - event.clientX;
        var y = this.top - event.clientY;
        this.offset = {
          x: x,
          y: y
        };
        this.dragging = 1;
        this.emit('drag-start');
      },
      focus: function focus(event) {
        if (this.drag) this.clearSelection();
        var vm = this;
        setTimeout(function () {
          if (!vm.clickButton) vm.emit('focus');
        }, 200);
      },
      center: function center() {
        var ww, wh;

        if (this.centered === 'window') {
          ww = window.innerWidth;
          wh = window.innerHeight;
        }

        if (this.centered === 'viewport') {
          var body = document.body;
          ww = body.clientWidth + body.scrollLeft;
          wh = body.clientHeight + body.scrollTop;
        }

        ww = ww || this.$parent.$el.clientWidth;
        wh = wh || this.$parent.$el.clientHeight;
        this.left = ww / 2 - this.$el.clientWidth / 2;
        this.top = wh / 2 - this.$el.clientHeight / 2;
      },
      setOptions: function setOptions(options) {
        if (options) {
          if (options.x) options.left = options.x;
          if (options.y) options.top = options.y;
          if (options.z) options.zIndex = options.z;
          this.drag = this.options.pinned ? false : this.drag; // available options

          var ops = this.availableOptions;
          var _iteratorNormalCompletion = true;
          var _didIteratorError = false;
          var _iteratorError = undefined;

          try {
            for (var _iterator = ops[Symbol.iterator](), _step; !(_iteratorNormalCompletion = (_step = _iterator.next()).done); _iteratorNormalCompletion = true) {
              var op = _step.value;

              if (this.options.hasOwnProperty(op)) {
                this.$set(this, op, this.options[op]);
              }
            }
          } catch (err) {
            _didIteratorError = true;
            _iteratorError = err;
          } finally {
            try {
              if (!_iteratorNormalCompletion && _iterator.return != null) {
                _iterator.return();
              }
            } finally {
              if (_didIteratorError) {
                throw _iteratorError;
              }
            }
          }
        }
      }
    }
  });
  // CONCATENATED MODULE: ./src/components/vue-dialog-drag.vue?vue&type=script&lang=js&
   /* harmony default export */ var components_vue_dialog_dragvue_type_script_lang_js_ = (vue_dialog_dragvue_type_script_lang_js_); 
  // EXTERNAL MODULE: ./src/components/vue-dialog-drag.vue?vue&type=style&index=0&lang=stylus&
  var vue_dialog_dragvue_type_style_index_0_lang_stylus_ = __webpack_require__("r8ud");

  // CONCATENATED MODULE: ./node_modules/vue-loader/lib/runtime/componentNormalizer.js
  /* globals __VUE_SSR_CONTEXT__ */

  // IMPORTANT: Do NOT use ES2015 features in this file (except for modules).
  // This module is a runtime utility for cleaner component module output and will
  // be included in the final webpack user bundle.

  function normalizeComponent (
    scriptExports,
    render,
    staticRenderFns,
    functionalTemplate,
    injectStyles,
    scopeId,
    moduleIdentifier, /* server only */
    shadowMode /* vue-cli only */
  ) {
    // Vue.extend constructor export interop
    var options = typeof scriptExports === 'function'
      ? scriptExports.options
      : scriptExports;

    // render functions
    if (render) {
      options.render = render;
      options.staticRenderFns = staticRenderFns;
      options._compiled = true;
    }

    // functional template
    if (functionalTemplate) {
      options.functional = true;
    }

    // scopedId
    if (scopeId) {
      options._scopeId = 'data-v-' + scopeId;
    }

    var hook;
    if (moduleIdentifier) { // server build
      hook = function (context) {
        // 2.3 injection
        context =
          context || // cached call
          (this.$vnode && this.$vnode.ssrContext) || // stateful
          (this.parent && this.parent.$vnode && this.parent.$vnode.ssrContext); // functional
        // 2.2 with runInNewContext: true
        if (!context && typeof __VUE_SSR_CONTEXT__ !== 'undefined') {
          context = __VUE_SSR_CONTEXT__;
        }
        // inject component styles
        if (injectStyles) {
          injectStyles.call(this, context);
        }
        // register component module identifier for async chunk inferrence
        if (context && context._registeredComponents) {
          context._registeredComponents.add(moduleIdentifier);
        }
      };
      // used by ssr in case component is cached and beforeCreate
      // never gets called
      options._ssrRegister = hook;
    } else if (injectStyles) {
      hook = shadowMode
        ? function () { injectStyles.call(this, this.$root.$options.shadowRoot); }
        : injectStyles;
    }

    if (hook) {
      if (options.functional) {
        // for template-only hot-reload because in that case the render fn doesn't
        // go through the normalizer
        options._injectStyles = hook;
        // register for functioal component in vue file
        var originalRender = options.render;
        options.render = function renderWithStyleInjection (h, context) {
          hook.call(context);
          return originalRender(h, context)
        };
      } else {
        // inject component registration as beforeCreate hook
        var existing = options.beforeCreate;
        options.beforeCreate = existing
          ? [].concat(existing, hook)
          : [hook];
      }
    }

    return {
      exports: scriptExports,
      options: options
    }
  }

  // CONCATENATED MODULE: ./src/components/vue-dialog-drag.vue






  /* normalize component */

  var component = normalizeComponent(
    components_vue_dialog_dragvue_type_script_lang_js_,
    render,
    staticRenderFns,
    false,
    null,
    null,
    null
    
  );

  /* harmony default export */ var vue_dialog_drag = (component.exports);
  // CONCATENATED MODULE: ./node_modules/@vue/cli-service/lib/commands/build/entry-lib.js


  /* harmony default export */ var entry_lib = __webpack_exports__["default"] = (vue_dialog_drag);



  /***/ }),

  /***/ "0/R4":
  /***/ (function(module, exports) {

  module.exports = function (it) {
    return typeof it === 'object' ? it !== null : typeof it === 'function';
  };


  /***/ }),

  /***/ "1MBn":
  /***/ (function(module, exports, __webpack_require__) {

  // all enumerable object keys, includes symbols
  var getKeys = __webpack_require__("DVgA");
  var gOPS = __webpack_require__("JiEa");
  var pIE = __webpack_require__("UqcF");
  module.exports = function (it) {
    var result = getKeys(it);
    var getSymbols = gOPS.f;
    if (getSymbols) {
      var symbols = getSymbols(it);
      var isEnum = pIE.f;
      var i = 0;
      var key;
      while (symbols.length > i) if (isEnum.call(it, key = symbols[i++])) result.push(key);
    } return result;
  };


  /***/ }),

  /***/ "1TsA":
  /***/ (function(module, exports) {

  module.exports = function (done, value) {
    return { value: value, done: !!done };
  };


  /***/ }),

  /***/ "2OiF":
  /***/ (function(module, exports) {

  module.exports = function (it) {
    if (typeof it != 'function') throw TypeError(it + ' is not a function!');
    return it;
  };


  /***/ }),

  /***/ "4R4u":
  /***/ (function(module, exports) {

  // IE 8- don't enum bug keys
  module.exports = (
    'constructor,hasOwnProperty,isPrototypeOf,propertyIsEnumerable,toLocaleString,toString,valueOf'
  ).split(',');


  /***/ }),

  /***/ "Afnz":
  /***/ (function(module, exports, __webpack_require__) {

  var LIBRARY = __webpack_require__("LQAc");
  var $export = __webpack_require__("XKFU");
  var redefine = __webpack_require__("KroJ");
  var hide = __webpack_require__("Mukb");
  var Iterators = __webpack_require__("hPIQ");
  var $iterCreate = __webpack_require__("QaDb");
  var setToStringTag = __webpack_require__("fyDq");
  var getPrototypeOf = __webpack_require__("OP3Y");
  var ITERATOR = __webpack_require__("K0xU")('iterator');
  var BUGGY = !([].keys && 'next' in [].keys()); // Safari has buggy iterators w/o `next`
  var FF_ITERATOR = '@@iterator';
  var KEYS = 'keys';
  var VALUES = 'values';

  var returnThis = function () { return this; };

  module.exports = function (Base, NAME, Constructor, next, DEFAULT, IS_SET, FORCED) {
    $iterCreate(Constructor, NAME, next);
    var getMethod = function (kind) {
      if (!BUGGY && kind in proto) return proto[kind];
      switch (kind) {
        case KEYS: return function keys() { return new Constructor(this, kind); };
        case VALUES: return function values() { return new Constructor(this, kind); };
      } return function entries() { return new Constructor(this, kind); };
    };
    var TAG = NAME + ' Iterator';
    var DEF_VALUES = DEFAULT == VALUES;
    var VALUES_BUG = false;
    var proto = Base.prototype;
    var $native = proto[ITERATOR] || proto[FF_ITERATOR] || DEFAULT && proto[DEFAULT];
    var $default = $native || getMethod(DEFAULT);
    var $entries = DEFAULT ? !DEF_VALUES ? $default : getMethod('entries') : undefined;
    var $anyNative = NAME == 'Array' ? proto.entries || $native : $native;
    var methods, key, IteratorPrototype;
    // Fix native
    if ($anyNative) {
      IteratorPrototype = getPrototypeOf($anyNative.call(new Base()));
      if (IteratorPrototype !== Object.prototype && IteratorPrototype.next) {
        // Set @@toStringTag to native iterators
        setToStringTag(IteratorPrototype, TAG, true);
        // fix for some old engines
        if (!LIBRARY && typeof IteratorPrototype[ITERATOR] != 'function') hide(IteratorPrototype, ITERATOR, returnThis);
      }
    }
    // fix Array#{values, @@iterator}.name in V8 / FF
    if (DEF_VALUES && $native && $native.name !== VALUES) {
      VALUES_BUG = true;
      $default = function values() { return $native.call(this); };
    }
    // Define iterator
    if ((!LIBRARY || FORCED) && (BUGGY || VALUES_BUG || !proto[ITERATOR])) {
      hide(proto, ITERATOR, $default);
    }
    // Plug for library
    Iterators[NAME] = $default;
    Iterators[TAG] = returnThis;
    if (DEFAULT) {
      methods = {
        values: DEF_VALUES ? $default : getMethod(VALUES),
        keys: IS_SET ? $default : getMethod(KEYS),
        entries: $entries
      };
      if (FORCED) for (key in methods) {
        if (!(key in proto)) redefine(proto, key, methods[key]);
      } else $export($export.P + $export.F * (BUGGY || VALUES_BUG), NAME, methods);
    }
    return methods;
  };


  /***/ }),

  /***/ "Ayid":
  /***/ (function(module, exports, __webpack_require__) {

  // extracted by mini-css-extract-plugin

  /***/ }),

  /***/ "DVgA":
  /***/ (function(module, exports, __webpack_require__) {

  // 19.1.2.14 / 15.2.3.14 Object.keys(O)
  var $keys = __webpack_require__("zhAb");
  var enumBugKeys = __webpack_require__("4R4u");

  module.exports = Object.keys || function keys(O) {
    return $keys(O, enumBugKeys);
  };


  /***/ }),

  /***/ "EWmC":
  /***/ (function(module, exports, __webpack_require__) {

  // 7.2.2 IsArray(argument)
  var cof = __webpack_require__("LZWt");
  module.exports = Array.isArray || function isArray(arg) {
    return cof(arg) == 'Array';
  };


  /***/ }),

  /***/ "EemH":
  /***/ (function(module, exports, __webpack_require__) {

  var pIE = __webpack_require__("UqcF");
  var createDesc = __webpack_require__("RjD/");
  var toIObject = __webpack_require__("aCFj");
  var toPrimitive = __webpack_require__("apmT");
  var has = __webpack_require__("aagx");
  var IE8_DOM_DEFINE = __webpack_require__("xpql");
  var gOPD = Object.getOwnPropertyDescriptor;

  exports.f = __webpack_require__("nh4g") ? gOPD : function getOwnPropertyDescriptor(O, P) {
    O = toIObject(O);
    P = toPrimitive(P, true);
    if (IE8_DOM_DEFINE) try {
      return gOPD(O, P);
    } catch (e) { /* empty */ }
    if (has(O, P)) return createDesc(!pIE.f.call(O, P), O[P]);
  };


  /***/ }),

  /***/ "FJW5":
  /***/ (function(module, exports, __webpack_require__) {

  var dP = __webpack_require__("hswa");
  var anObject = __webpack_require__("y3w9");
  var getKeys = __webpack_require__("DVgA");

  module.exports = __webpack_require__("nh4g") ? Object.defineProperties : function defineProperties(O, Properties) {
    anObject(O);
    var keys = getKeys(Properties);
    var length = keys.length;
    var i = 0;
    var P;
    while (length > i) dP.f(O, P = keys[i++], Properties[P]);
    return O;
  };


  /***/ }),

  /***/ "HrLf":
  /***/ (function(module, exports, __webpack_require__) {

  // This file is imported into lib/wc client bundles.

  if (typeof window !== 'undefined') {
    var i;
    if ((i = window.document.currentScript) && (i = i.src.match(/(.+\/)[^/]+\.js$/))) {
      __webpack_require__.p = i[1]; // eslint-disable-line
    }
  }


  /***/ }),

  /***/ "Iw71":
  /***/ (function(module, exports, __webpack_require__) {

  var isObject = __webpack_require__("0/R4");
  var document = __webpack_require__("dyZX").document;
  // typeof document.createElement is 'object' in old IE
  var is = isObject(document) && isObject(document.createElement);
  module.exports = function (it) {
    return is ? document.createElement(it) : {};
  };


  /***/ }),

  /***/ "JiEa":
  /***/ (function(module, exports) {

  exports.f = Object.getOwnPropertySymbols;


  /***/ }),

  /***/ "K0xU":
  /***/ (function(module, exports, __webpack_require__) {

  var store = __webpack_require__("VTer")('wks');
  var uid = __webpack_require__("ylqs");
  var Symbol = __webpack_require__("dyZX").Symbol;
  var USE_SYMBOL = typeof Symbol == 'function';

  var $exports = module.exports = function (name) {
    return store[name] || (store[name] =
      USE_SYMBOL && Symbol[name] || (USE_SYMBOL ? Symbol : uid)('Symbol.' + name));
  };

  $exports.store = store;


  /***/ }),

  /***/ "KroJ":
  /***/ (function(module, exports, __webpack_require__) {

  var global = __webpack_require__("dyZX");
  var hide = __webpack_require__("Mukb");
  var has = __webpack_require__("aagx");
  var SRC = __webpack_require__("ylqs")('src');
  var TO_STRING = 'toString';
  var $toString = Function[TO_STRING];
  var TPL = ('' + $toString).split(TO_STRING);

  __webpack_require__("g3g5").inspectSource = function (it) {
    return $toString.call(it);
  };

  (module.exports = function (O, key, val, safe) {
    var isFunction = typeof val == 'function';
    if (isFunction) has(val, 'name') || hide(val, 'name', key);
    if (O[key] === val) return;
    if (isFunction) has(val, SRC) || hide(val, SRC, O[key] ? '' + O[key] : TPL.join(String(key)));
    if (O === global) {
      O[key] = val;
    } else if (!safe) {
      delete O[key];
      hide(O, key, val);
    } else if (O[key]) {
      O[key] = val;
    } else {
      hide(O, key, val);
    }
  // add fake Function#toString for correct work wrapped methods / constructors with methods like LoDash isNative
  })(Function.prototype, TO_STRING, function toString() {
    return typeof this == 'function' && this[SRC] || $toString.call(this);
  });


  /***/ }),

  /***/ "Kuth":
  /***/ (function(module, exports, __webpack_require__) {

  // 19.1.2.2 / 15.2.3.5 Object.create(O [, Properties])
  var anObject = __webpack_require__("y3w9");
  var dPs = __webpack_require__("FJW5");
  var enumBugKeys = __webpack_require__("4R4u");
  var IE_PROTO = __webpack_require__("YTvA")('IE_PROTO');
  var Empty = function () { /* empty */ };
  var PROTOTYPE = 'prototype';

  // Create object with fake `null` prototype: use iframe Object with cleared prototype
  var createDict = function () {
    // Thrash, waste and sodomy: IE GC bug
    var iframe = __webpack_require__("Iw71")('iframe');
    var i = enumBugKeys.length;
    var lt = '<';
    var gt = '>';
    var iframeDocument;
    iframe.style.display = 'none';
    __webpack_require__("+rLv").appendChild(iframe);
    iframe.src = 'javascript:'; // eslint-disable-line no-script-url
    // createDict = iframe.contentWindow.Object;
    // html.removeChild(iframe);
    iframeDocument = iframe.contentWindow.document;
    iframeDocument.open();
    iframeDocument.write(lt + 'script' + gt + 'document.F=Object' + lt + '/script' + gt);
    iframeDocument.close();
    createDict = iframeDocument.F;
    while (i--) delete createDict[PROTOTYPE][enumBugKeys[i]];
    return createDict();
  };

  module.exports = Object.create || function create(O, Properties) {
    var result;
    if (O !== null) {
      Empty[PROTOTYPE] = anObject(O);
      result = new Empty();
      Empty[PROTOTYPE] = null;
      // add "__proto__" for Object.getPrototypeOf polyfill
      result[IE_PROTO] = O;
    } else result = createDict();
    return Properties === undefined ? result : dPs(result, Properties);
  };


  /***/ }),

  /***/ "LQAc":
  /***/ (function(module, exports) {

  module.exports = false;


  /***/ }),

  /***/ "LZWt":
  /***/ (function(module, exports) {

  var toString = {}.toString;

  module.exports = function (it) {
    return toString.call(it).slice(8, -1);
  };


  /***/ }),

  /***/ "Mukb":
  /***/ (function(module, exports, __webpack_require__) {

  var dP = __webpack_require__("hswa");
  var createDesc = __webpack_require__("RjD/");
  module.exports = __webpack_require__("nh4g") ? function (object, key, value) {
    return dP.f(object, key, createDesc(1, value));
  } : function (object, key, value) {
    object[key] = value;
    return object;
  };


  /***/ }),

  /***/ "N8g3":
  /***/ (function(module, exports, __webpack_require__) {

  exports.f = __webpack_require__("K0xU");


  /***/ }),

  /***/ "OP3Y":
  /***/ (function(module, exports, __webpack_require__) {

  // 19.1.2.9 / 15.2.3.2 Object.getPrototypeOf(O)
  var has = __webpack_require__("aagx");
  var toObject = __webpack_require__("S/j/");
  var IE_PROTO = __webpack_require__("YTvA")('IE_PROTO');
  var ObjectProto = Object.prototype;

  module.exports = Object.getPrototypeOf || function (O) {
    O = toObject(O);
    if (has(O, IE_PROTO)) return O[IE_PROTO];
    if (typeof O.constructor == 'function' && O instanceof O.constructor) {
      return O.constructor.prototype;
    } return O instanceof Object ? ObjectProto : null;
  };


  /***/ }),

  /***/ "OnI7":
  /***/ (function(module, exports, __webpack_require__) {

  var global = __webpack_require__("dyZX");
  var core = __webpack_require__("g3g5");
  var LIBRARY = __webpack_require__("LQAc");
  var wksExt = __webpack_require__("N8g3");
  var defineProperty = __webpack_require__("hswa").f;
  module.exports = function (name) {
    var $Symbol = core.Symbol || (core.Symbol = LIBRARY ? {} : global.Symbol || {});
    if (name.charAt(0) != '_' && !(name in $Symbol)) defineProperty($Symbol, name, { value: wksExt.f(name) });
  };


  /***/ }),

  /***/ "QaDb":
  /***/ (function(module, exports, __webpack_require__) {

  var create = __webpack_require__("Kuth");
  var descriptor = __webpack_require__("RjD/");
  var setToStringTag = __webpack_require__("fyDq");
  var IteratorPrototype = {};

  // 25.1.2.1.1 %IteratorPrototype%[@@iterator]()
  __webpack_require__("Mukb")(IteratorPrototype, __webpack_require__("K0xU")('iterator'), function () { return this; });

  module.exports = function (Constructor, NAME, next) {
    Constructor.prototype = create(IteratorPrototype, { next: descriptor(1, next) });
    setToStringTag(Constructor, NAME + ' Iterator');
  };


  /***/ }),

  /***/ "RYi7":
  /***/ (function(module, exports) {

  // 7.1.4 ToInteger
  var ceil = Math.ceil;
  var floor = Math.floor;
  module.exports = function (it) {
    return isNaN(it = +it) ? 0 : (it > 0 ? floor : ceil)(it);
  };


  /***/ }),

  /***/ "RjD/":
  /***/ (function(module, exports) {

  module.exports = function (bitmap, value) {
    return {
      enumerable: !(bitmap & 1),
      configurable: !(bitmap & 2),
      writable: !(bitmap & 4),
      value: value
    };
  };


  /***/ }),

  /***/ "S/j/":
  /***/ (function(module, exports, __webpack_require__) {

  // 7.1.13 ToObject(argument)
  var defined = __webpack_require__("vhPU");
  module.exports = function (it) {
    return Object(defined(it));
  };


  /***/ }),

  /***/ "UqcF":
  /***/ (function(module, exports) {

  exports.f = {}.propertyIsEnumerable;


  /***/ }),

  /***/ "VTer":
  /***/ (function(module, exports, __webpack_require__) {

  var core = __webpack_require__("g3g5");
  var global = __webpack_require__("dyZX");
  var SHARED = '__core-js_shared__';
  var store = global[SHARED] || (global[SHARED] = {});

  (module.exports = function (key, value) {
    return store[key] || (store[key] = value !== undefined ? value : {});
  })('versions', []).push({
    version: core.version,
    mode: __webpack_require__("LQAc") ? 'pure' : 'global',
    copyright: '© 2018 Denis Pushkarev (zloirock.ru)'
  });


  /***/ }),

  /***/ "XKFU":
  /***/ (function(module, exports, __webpack_require__) {

  var global = __webpack_require__("dyZX");
  var core = __webpack_require__("g3g5");
  var hide = __webpack_require__("Mukb");
  var redefine = __webpack_require__("KroJ");
  var ctx = __webpack_require__("m0Pp");
  var PROTOTYPE = 'prototype';

  var $export = function (type, name, source) {
    var IS_FORCED = type & $export.F;
    var IS_GLOBAL = type & $export.G;
    var IS_STATIC = type & $export.S;
    var IS_PROTO = type & $export.P;
    var IS_BIND = type & $export.B;
    var target = IS_GLOBAL ? global : IS_STATIC ? global[name] || (global[name] = {}) : (global[name] || {})[PROTOTYPE];
    var exports = IS_GLOBAL ? core : core[name] || (core[name] = {});
    var expProto = exports[PROTOTYPE] || (exports[PROTOTYPE] = {});
    var key, own, out, exp;
    if (IS_GLOBAL) source = name;
    for (key in source) {
      // contains in native
      own = !IS_FORCED && target && target[key] !== undefined;
      // export native or passed
      out = (own ? target : source)[key];
      // bind timers to global for call from export context
      exp = IS_BIND && own ? ctx(out, global) : IS_PROTO && typeof out == 'function' ? ctx(Function.call, out) : out;
      // extend global
      if (target) redefine(target, key, out, type & $export.U);
      // export
      if (exports[key] != out) hide(exports, key, exp);
      if (IS_PROTO && expProto[key] != out) expProto[key] = out;
    }
  };
  global.core = core;
  // type bitmap
  $export.F = 1;   // forced
  $export.G = 2;   // global
  $export.S = 4;   // static
  $export.P = 8;   // proto
  $export.B = 16;  // bind
  $export.W = 32;  // wrap
  $export.U = 64;  // safe
  $export.R = 128; // real proto method for `library`
  module.exports = $export;


  /***/ }),

  /***/ "YTvA":
  /***/ (function(module, exports, __webpack_require__) {

  var shared = __webpack_require__("VTer")('keys');
  var uid = __webpack_require__("ylqs");
  module.exports = function (key) {
    return shared[key] || (shared[key] = uid(key));
  };


  /***/ }),

  /***/ "Ymqv":
  /***/ (function(module, exports, __webpack_require__) {

  // fallback for non-array-like ES3 and non-enumerable old V8 strings
  var cof = __webpack_require__("LZWt");
  // eslint-disable-next-line no-prototype-builtins
  module.exports = Object('z').propertyIsEnumerable(0) ? Object : function (it) {
    return cof(it) == 'String' ? it.split('') : Object(it);
  };


  /***/ }),

  /***/ "Z6vF":
  /***/ (function(module, exports, __webpack_require__) {

  var META = __webpack_require__("ylqs")('meta');
  var isObject = __webpack_require__("0/R4");
  var has = __webpack_require__("aagx");
  var setDesc = __webpack_require__("hswa").f;
  var id = 0;
  var isExtensible = Object.isExtensible || function () {
    return true;
  };
  var FREEZE = !__webpack_require__("eeVq")(function () {
    return isExtensible(Object.preventExtensions({}));
  });
  var setMeta = function (it) {
    setDesc(it, META, { value: {
      i: 'O' + ++id, // object ID
      w: {}          // weak collections IDs
    } });
  };
  var fastKey = function (it, create) {
    // return primitive with prefix
    if (!isObject(it)) return typeof it == 'symbol' ? it : (typeof it == 'string' ? 'S' : 'P') + it;
    if (!has(it, META)) {
      // can't set metadata to uncaught frozen object
      if (!isExtensible(it)) return 'F';
      // not necessary to add metadata
      if (!create) return 'E';
      // add missing metadata
      setMeta(it);
    // return object ID
    } return it[META].i;
  };
  var getWeak = function (it, create) {
    if (!has(it, META)) {
      // can't set metadata to uncaught frozen object
      if (!isExtensible(it)) return true;
      // not necessary to add metadata
      if (!create) return false;
      // add missing metadata
      setMeta(it);
    // return hash weak collections IDs
    } return it[META].w;
  };
  // add metadata on freeze-family methods calling
  var onFreeze = function (it) {
    if (FREEZE && meta.NEED && isExtensible(it) && !has(it, META)) setMeta(it);
    return it;
  };
  var meta = module.exports = {
    KEY: META,
    NEED: false,
    fastKey: fastKey,
    getWeak: getWeak,
    onFreeze: onFreeze
  };


  /***/ }),

  /***/ "aCFj":
  /***/ (function(module, exports, __webpack_require__) {

  // to indexed object, toObject with fallback for non-array-like ES3 strings
  var IObject = __webpack_require__("Ymqv");
  var defined = __webpack_require__("vhPU");
  module.exports = function (it) {
    return IObject(defined(it));
  };


  /***/ }),

  /***/ "aagx":
  /***/ (function(module, exports) {

  var hasOwnProperty = {}.hasOwnProperty;
  module.exports = function (it, key) {
    return hasOwnProperty.call(it, key);
  };


  /***/ }),

  /***/ "apmT":
  /***/ (function(module, exports, __webpack_require__) {

  // 7.1.1 ToPrimitive(input [, PreferredType])
  var isObject = __webpack_require__("0/R4");
  // instead of the ES6 spec version, we didn't implement @@toPrimitive case
  // and the second argument - flag - preferred type is a string
  module.exports = function (it, S) {
    if (!isObject(it)) return it;
    var fn, val;
    if (S && typeof (fn = it.toString) == 'function' && !isObject(val = fn.call(it))) return val;
    if (typeof (fn = it.valueOf) == 'function' && !isObject(val = fn.call(it))) return val;
    if (!S && typeof (fn = it.toString) == 'function' && !isObject(val = fn.call(it))) return val;
    throw TypeError("Can't convert object to primitive value");
  };


  /***/ }),

  /***/ "d/Gc":
  /***/ (function(module, exports, __webpack_require__) {

  var toInteger = __webpack_require__("RYi7");
  var max = Math.max;
  var min = Math.min;
  module.exports = function (index, length) {
    index = toInteger(index);
    return index < 0 ? max(index + length, 0) : min(index, length);
  };


  /***/ }),

  /***/ "dyZX":
  /***/ (function(module, exports) {

  // https://github.com/zloirock/core-js/issues/86#issuecomment-115759028
  var global = module.exports = typeof window != 'undefined' && window.Math == Math
    ? window : typeof self != 'undefined' && self.Math == Math ? self
    // eslint-disable-next-line no-new-func
    : Function('return this')();
  if (typeof __g == 'number') __g = global; // eslint-disable-line no-undef


  /***/ }),

  /***/ "e7yV":
  /***/ (function(module, exports, __webpack_require__) {

  // fallback for IE11 buggy Object.getOwnPropertyNames with iframe and window
  var toIObject = __webpack_require__("aCFj");
  var gOPN = __webpack_require__("kJMx").f;
  var toString = {}.toString;

  var windowNames = typeof window == 'object' && window && Object.getOwnPropertyNames
    ? Object.getOwnPropertyNames(window) : [];

  var getWindowNames = function (it) {
    try {
      return gOPN(it);
    } catch (e) {
      return windowNames.slice();
    }
  };

  module.exports.f = function getOwnPropertyNames(it) {
    return windowNames && toString.call(it) == '[object Window]' ? getWindowNames(it) : gOPN(toIObject(it));
  };


  /***/ }),

  /***/ "eeVq":
  /***/ (function(module, exports) {

  module.exports = function (exec) {
    try {
      return !!exec();
    } catch (e) {
      return true;
    }
  };


  /***/ }),

  /***/ "fyDq":
  /***/ (function(module, exports, __webpack_require__) {

  var def = __webpack_require__("hswa").f;
  var has = __webpack_require__("aagx");
  var TAG = __webpack_require__("K0xU")('toStringTag');

  module.exports = function (it, tag, stat) {
    if (it && !has(it = stat ? it : it.prototype, TAG)) def(it, TAG, { configurable: true, value: tag });
  };


  /***/ }),

  /***/ "g3g5":
  /***/ (function(module, exports) {

  var core = module.exports = { version: '2.5.7' };
  if (typeof __e == 'number') __e = core; // eslint-disable-line no-undef


  /***/ }),

  /***/ "hPIQ":
  /***/ (function(module, exports) {

  module.exports = {};


  /***/ }),

  /***/ "hswa":
  /***/ (function(module, exports, __webpack_require__) {

  var anObject = __webpack_require__("y3w9");
  var IE8_DOM_DEFINE = __webpack_require__("xpql");
  var toPrimitive = __webpack_require__("apmT");
  var dP = Object.defineProperty;

  exports.f = __webpack_require__("nh4g") ? Object.defineProperty : function defineProperty(O, P, Attributes) {
    anObject(O);
    P = toPrimitive(P, true);
    anObject(Attributes);
    if (IE8_DOM_DEFINE) try {
      return dP(O, P, Attributes);
    } catch (e) { /* empty */ }
    if ('get' in Attributes || 'set' in Attributes) throw TypeError('Accessors not supported!');
    if ('value' in Attributes) O[P] = Attributes.value;
    return O;
  };


  /***/ }),

  /***/ "ioFf":
  /***/ (function(module, exports, __webpack_require__) {

  // ECMAScript 6 symbols shim
  var global = __webpack_require__("dyZX");
  var has = __webpack_require__("aagx");
  var DESCRIPTORS = __webpack_require__("nh4g");
  var $export = __webpack_require__("XKFU");
  var redefine = __webpack_require__("KroJ");
  var META = __webpack_require__("Z6vF").KEY;
  var $fails = __webpack_require__("eeVq");
  var shared = __webpack_require__("VTer");
  var setToStringTag = __webpack_require__("fyDq");
  var uid = __webpack_require__("ylqs");
  var wks = __webpack_require__("K0xU");
  var wksExt = __webpack_require__("N8g3");
  var wksDefine = __webpack_require__("OnI7");
  var enumKeys = __webpack_require__("1MBn");
  var isArray = __webpack_require__("EWmC");
  var anObject = __webpack_require__("y3w9");
  var isObject = __webpack_require__("0/R4");
  var toIObject = __webpack_require__("aCFj");
  var toPrimitive = __webpack_require__("apmT");
  var createDesc = __webpack_require__("RjD/");
  var _create = __webpack_require__("Kuth");
  var gOPNExt = __webpack_require__("e7yV");
  var $GOPD = __webpack_require__("EemH");
  var $DP = __webpack_require__("hswa");
  var $keys = __webpack_require__("DVgA");
  var gOPD = $GOPD.f;
  var dP = $DP.f;
  var gOPN = gOPNExt.f;
  var $Symbol = global.Symbol;
  var $JSON = global.JSON;
  var _stringify = $JSON && $JSON.stringify;
  var PROTOTYPE = 'prototype';
  var HIDDEN = wks('_hidden');
  var TO_PRIMITIVE = wks('toPrimitive');
  var isEnum = {}.propertyIsEnumerable;
  var SymbolRegistry = shared('symbol-registry');
  var AllSymbols = shared('symbols');
  var OPSymbols = shared('op-symbols');
  var ObjectProto = Object[PROTOTYPE];
  var USE_NATIVE = typeof $Symbol == 'function';
  var QObject = global.QObject;
  // Don't use setters in Qt Script, https://github.com/zloirock/core-js/issues/173
  var setter = !QObject || !QObject[PROTOTYPE] || !QObject[PROTOTYPE].findChild;

  // fallback for old Android, https://code.google.com/p/v8/issues/detail?id=687
  var setSymbolDesc = DESCRIPTORS && $fails(function () {
    return _create(dP({}, 'a', {
      get: function () { return dP(this, 'a', { value: 7 }).a; }
    })).a != 7;
  }) ? function (it, key, D) {
    var protoDesc = gOPD(ObjectProto, key);
    if (protoDesc) delete ObjectProto[key];
    dP(it, key, D);
    if (protoDesc && it !== ObjectProto) dP(ObjectProto, key, protoDesc);
  } : dP;

  var wrap = function (tag) {
    var sym = AllSymbols[tag] = _create($Symbol[PROTOTYPE]);
    sym._k = tag;
    return sym;
  };

  var isSymbol = USE_NATIVE && typeof $Symbol.iterator == 'symbol' ? function (it) {
    return typeof it == 'symbol';
  } : function (it) {
    return it instanceof $Symbol;
  };

  var $defineProperty = function defineProperty(it, key, D) {
    if (it === ObjectProto) $defineProperty(OPSymbols, key, D);
    anObject(it);
    key = toPrimitive(key, true);
    anObject(D);
    if (has(AllSymbols, key)) {
      if (!D.enumerable) {
        if (!has(it, HIDDEN)) dP(it, HIDDEN, createDesc(1, {}));
        it[HIDDEN][key] = true;
      } else {
        if (has(it, HIDDEN) && it[HIDDEN][key]) it[HIDDEN][key] = false;
        D = _create(D, { enumerable: createDesc(0, false) });
      } return setSymbolDesc(it, key, D);
    } return dP(it, key, D);
  };
  var $defineProperties = function defineProperties(it, P) {
    anObject(it);
    var keys = enumKeys(P = toIObject(P));
    var i = 0;
    var l = keys.length;
    var key;
    while (l > i) $defineProperty(it, key = keys[i++], P[key]);
    return it;
  };
  var $create = function create(it, P) {
    return P === undefined ? _create(it) : $defineProperties(_create(it), P);
  };
  var $propertyIsEnumerable = function propertyIsEnumerable(key) {
    var E = isEnum.call(this, key = toPrimitive(key, true));
    if (this === ObjectProto && has(AllSymbols, key) && !has(OPSymbols, key)) return false;
    return E || !has(this, key) || !has(AllSymbols, key) || has(this, HIDDEN) && this[HIDDEN][key] ? E : true;
  };
  var $getOwnPropertyDescriptor = function getOwnPropertyDescriptor(it, key) {
    it = toIObject(it);
    key = toPrimitive(key, true);
    if (it === ObjectProto && has(AllSymbols, key) && !has(OPSymbols, key)) return;
    var D = gOPD(it, key);
    if (D && has(AllSymbols, key) && !(has(it, HIDDEN) && it[HIDDEN][key])) D.enumerable = true;
    return D;
  };
  var $getOwnPropertyNames = function getOwnPropertyNames(it) {
    var names = gOPN(toIObject(it));
    var result = [];
    var i = 0;
    var key;
    while (names.length > i) {
      if (!has(AllSymbols, key = names[i++]) && key != HIDDEN && key != META) result.push(key);
    } return result;
  };
  var $getOwnPropertySymbols = function getOwnPropertySymbols(it) {
    var IS_OP = it === ObjectProto;
    var names = gOPN(IS_OP ? OPSymbols : toIObject(it));
    var result = [];
    var i = 0;
    var key;
    while (names.length > i) {
      if (has(AllSymbols, key = names[i++]) && (IS_OP ? has(ObjectProto, key) : true)) result.push(AllSymbols[key]);
    } return result;
  };

  // 19.4.1.1 Symbol([description])
  if (!USE_NATIVE) {
    $Symbol = function Symbol() {
      if (this instanceof $Symbol) throw TypeError('Symbol is not a constructor!');
      var tag = uid(arguments.length > 0 ? arguments[0] : undefined);
      var $set = function (value) {
        if (this === ObjectProto) $set.call(OPSymbols, value);
        if (has(this, HIDDEN) && has(this[HIDDEN], tag)) this[HIDDEN][tag] = false;
        setSymbolDesc(this, tag, createDesc(1, value));
      };
      if (DESCRIPTORS && setter) setSymbolDesc(ObjectProto, tag, { configurable: true, set: $set });
      return wrap(tag);
    };
    redefine($Symbol[PROTOTYPE], 'toString', function toString() {
      return this._k;
    });

    $GOPD.f = $getOwnPropertyDescriptor;
    $DP.f = $defineProperty;
    __webpack_require__("kJMx").f = gOPNExt.f = $getOwnPropertyNames;
    __webpack_require__("UqcF").f = $propertyIsEnumerable;
    __webpack_require__("JiEa").f = $getOwnPropertySymbols;

    if (DESCRIPTORS && !__webpack_require__("LQAc")) {
      redefine(ObjectProto, 'propertyIsEnumerable', $propertyIsEnumerable, true);
    }

    wksExt.f = function (name) {
      return wrap(wks(name));
    };
  }

  $export($export.G + $export.W + $export.F * !USE_NATIVE, { Symbol: $Symbol });

  for (var es6Symbols = (
    // 19.4.2.2, 19.4.2.3, 19.4.2.4, 19.4.2.6, 19.4.2.8, 19.4.2.9, 19.4.2.10, 19.4.2.11, 19.4.2.12, 19.4.2.13, 19.4.2.14
    'hasInstance,isConcatSpreadable,iterator,match,replace,search,species,split,toPrimitive,toStringTag,unscopables'
  ).split(','), j = 0; es6Symbols.length > j;)wks(es6Symbols[j++]);

  for (var wellKnownSymbols = $keys(wks.store), k = 0; wellKnownSymbols.length > k;) wksDefine(wellKnownSymbols[k++]);

  $export($export.S + $export.F * !USE_NATIVE, 'Symbol', {
    // 19.4.2.1 Symbol.for(key)
    'for': function (key) {
      return has(SymbolRegistry, key += '')
        ? SymbolRegistry[key]
        : SymbolRegistry[key] = $Symbol(key);
    },
    // 19.4.2.5 Symbol.keyFor(sym)
    keyFor: function keyFor(sym) {
      if (!isSymbol(sym)) throw TypeError(sym + ' is not a symbol!');
      for (var key in SymbolRegistry) if (SymbolRegistry[key] === sym) return key;
    },
    useSetter: function () { setter = true; },
    useSimple: function () { setter = false; }
  });

  $export($export.S + $export.F * !USE_NATIVE, 'Object', {
    // 19.1.2.2 Object.create(O [, Properties])
    create: $create,
    // 19.1.2.4 Object.defineProperty(O, P, Attributes)
    defineProperty: $defineProperty,
    // 19.1.2.3 Object.defineProperties(O, Properties)
    defineProperties: $defineProperties,
    // 19.1.2.6 Object.getOwnPropertyDescriptor(O, P)
    getOwnPropertyDescriptor: $getOwnPropertyDescriptor,
    // 19.1.2.7 Object.getOwnPropertyNames(O)
    getOwnPropertyNames: $getOwnPropertyNames,
    // 19.1.2.8 Object.getOwnPropertySymbols(O)
    getOwnPropertySymbols: $getOwnPropertySymbols
  });

  // 24.3.2 JSON.stringify(value [, replacer [, space]])
  $JSON && $export($export.S + $export.F * (!USE_NATIVE || $fails(function () {
    var S = $Symbol();
    // MS Edge converts symbol values to JSON as {}
    // WebKit converts symbol values to JSON as null
    // V8 throws on boxed symbols
    return _stringify([S]) != '[null]' || _stringify({ a: S }) != '{}' || _stringify(Object(S)) != '{}';
  })), 'JSON', {
    stringify: function stringify(it) {
      var args = [it];
      var i = 1;
      var replacer, $replacer;
      while (arguments.length > i) args.push(arguments[i++]);
      $replacer = replacer = args[1];
      if (!isObject(replacer) && it === undefined || isSymbol(it)) return; // IE8 returns string on undefined
      if (!isArray(replacer)) replacer = function (key, value) {
        if (typeof $replacer == 'function') value = $replacer.call(this, key, value);
        if (!isSymbol(value)) return value;
      };
      args[1] = replacer;
      return _stringify.apply($JSON, args);
    }
  });

  // 19.4.3.4 Symbol.prototype[@@toPrimitive](hint)
  $Symbol[PROTOTYPE][TO_PRIMITIVE] || __webpack_require__("Mukb")($Symbol[PROTOTYPE], TO_PRIMITIVE, $Symbol[PROTOTYPE].valueOf);
  // 19.4.3.5 Symbol.prototype[@@toStringTag]
  setToStringTag($Symbol, 'Symbol');
  // 20.2.1.9 Math[@@toStringTag]
  setToStringTag(Math, 'Math', true);
  // 24.3.3 JSON[@@toStringTag]
  setToStringTag(global.JSON, 'JSON', true);


  /***/ }),

  /***/ "kJMx":
  /***/ (function(module, exports, __webpack_require__) {

  // 19.1.2.7 / 15.2.3.4 Object.getOwnPropertyNames(O)
  var $keys = __webpack_require__("zhAb");
  var hiddenKeys = __webpack_require__("4R4u").concat('length', 'prototype');

  exports.f = Object.getOwnPropertyNames || function getOwnPropertyNames(O) {
    return $keys(O, hiddenKeys);
  };


  /***/ }),

  /***/ "m0Pp":
  /***/ (function(module, exports, __webpack_require__) {

  // optional / simple context binding
  var aFunction = __webpack_require__("2OiF");
  module.exports = function (fn, that, length) {
    aFunction(fn);
    if (that === undefined) return fn;
    switch (length) {
      case 1: return function (a) {
        return fn.call(that, a);
      };
      case 2: return function (a, b) {
        return fn.call(that, a, b);
      };
      case 3: return function (a, b, c) {
        return fn.call(that, a, b, c);
      };
    }
    return function (/* ...args */) {
      return fn.apply(that, arguments);
    };
  };


  /***/ }),

  /***/ "nGyu":
  /***/ (function(module, exports, __webpack_require__) {

  // 22.1.3.31 Array.prototype[@@unscopables]
  var UNSCOPABLES = __webpack_require__("K0xU")('unscopables');
  var ArrayProto = Array.prototype;
  if (ArrayProto[UNSCOPABLES] == undefined) __webpack_require__("Mukb")(ArrayProto, UNSCOPABLES, {});
  module.exports = function (key) {
    ArrayProto[UNSCOPABLES][key] = true;
  };


  /***/ }),

  /***/ "ne8i":
  /***/ (function(module, exports, __webpack_require__) {

  // 7.1.15 ToLength
  var toInteger = __webpack_require__("RYi7");
  var min = Math.min;
  module.exports = function (it) {
    return it > 0 ? min(toInteger(it), 0x1fffffffffffff) : 0; // pow(2, 53) - 1 == 9007199254740991
  };


  /***/ }),

  /***/ "nh4g":
  /***/ (function(module, exports, __webpack_require__) {

  // Thank's IE8 for his funny defineProperty
  module.exports = !__webpack_require__("eeVq")(function () {
    return Object.defineProperty({}, 'a', { get: function () { return 7; } }).a != 7;
  });


  /***/ }),

  /***/ "r8ud":
  /***/ (function(module, __webpack_exports__, __webpack_require__) {
  /* harmony import */ var _node_modules_mini_css_extract_plugin_dist_loader_js_node_modules_css_loader_index_js_ref_11_oneOf_1_1_node_modules_vue_loader_lib_loaders_stylePostLoader_js_node_modules_postcss_loader_lib_index_js_ref_11_oneOf_1_2_node_modules_stylus_loader_index_js_ref_11_oneOf_1_3_node_modules_cache_loader_dist_cjs_js_ref_0_0_node_modules_vue_loader_lib_index_js_vue_loader_options_vue_dialog_drag_vue_vue_type_style_index_0_lang_stylus___WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__("Ayid");
  /* harmony import */ var _node_modules_mini_css_extract_plugin_dist_loader_js_node_modules_css_loader_index_js_ref_11_oneOf_1_1_node_modules_vue_loader_lib_loaders_stylePostLoader_js_node_modules_postcss_loader_lib_index_js_ref_11_oneOf_1_2_node_modules_stylus_loader_index_js_ref_11_oneOf_1_3_node_modules_cache_loader_dist_cjs_js_ref_0_0_node_modules_vue_loader_lib_index_js_vue_loader_options_vue_dialog_drag_vue_vue_type_style_index_0_lang_stylus___WEBPACK_IMPORTED_MODULE_0___default = /*#__PURE__*/__webpack_require__.n(_node_modules_mini_css_extract_plugin_dist_loader_js_node_modules_css_loader_index_js_ref_11_oneOf_1_1_node_modules_vue_loader_lib_loaders_stylePostLoader_js_node_modules_postcss_loader_lib_index_js_ref_11_oneOf_1_2_node_modules_stylus_loader_index_js_ref_11_oneOf_1_3_node_modules_cache_loader_dist_cjs_js_ref_0_0_node_modules_vue_loader_lib_index_js_vue_loader_options_vue_dialog_drag_vue_vue_type_style_index_0_lang_stylus___WEBPACK_IMPORTED_MODULE_0__);
  /* unused harmony reexport * */
   /* unused harmony default export */ var _unused_webpack_default_export = (_node_modules_mini_css_extract_plugin_dist_loader_js_node_modules_css_loader_index_js_ref_11_oneOf_1_1_node_modules_vue_loader_lib_loaders_stylePostLoader_js_node_modules_postcss_loader_lib_index_js_ref_11_oneOf_1_2_node_modules_stylus_loader_index_js_ref_11_oneOf_1_3_node_modules_cache_loader_dist_cjs_js_ref_0_0_node_modules_vue_loader_lib_index_js_vue_loader_options_vue_dialog_drag_vue_vue_type_style_index_0_lang_stylus___WEBPACK_IMPORTED_MODULE_0___default.a); 

  /***/ }),

  /***/ "rE2o":
  /***/ (function(module, exports, __webpack_require__) {

  __webpack_require__("OnI7")('asyncIterator');


  /***/ }),

  /***/ "rGqo":
  /***/ (function(module, exports, __webpack_require__) {

  var $iterators = __webpack_require__("yt8O");
  var getKeys = __webpack_require__("DVgA");
  var redefine = __webpack_require__("KroJ");
  var global = __webpack_require__("dyZX");
  var hide = __webpack_require__("Mukb");
  var Iterators = __webpack_require__("hPIQ");
  var wks = __webpack_require__("K0xU");
  var ITERATOR = wks('iterator');
  var TO_STRING_TAG = wks('toStringTag');
  var ArrayValues = Iterators.Array;

  var DOMIterables = {
    CSSRuleList: true, // TODO: Not spec compliant, should be false.
    CSSStyleDeclaration: false,
    CSSValueList: false,
    ClientRectList: false,
    DOMRectList: false,
    DOMStringList: false,
    DOMTokenList: true,
    DataTransferItemList: false,
    FileList: false,
    HTMLAllCollection: false,
    HTMLCollection: false,
    HTMLFormElement: false,
    HTMLSelectElement: false,
    MediaList: true, // TODO: Not spec compliant, should be false.
    MimeTypeArray: false,
    NamedNodeMap: false,
    NodeList: true,
    PaintRequestList: false,
    Plugin: false,
    PluginArray: false,
    SVGLengthList: false,
    SVGNumberList: false,
    SVGPathSegList: false,
    SVGPointList: false,
    SVGStringList: false,
    SVGTransformList: false,
    SourceBufferList: false,
    StyleSheetList: true, // TODO: Not spec compliant, should be false.
    TextTrackCueList: false,
    TextTrackList: false,
    TouchList: false
  };

  for (var collections = getKeys(DOMIterables), i = 0; i < collections.length; i++) {
    var NAME = collections[i];
    var explicit = DOMIterables[NAME];
    var Collection = global[NAME];
    var proto = Collection && Collection.prototype;
    var key;
    if (proto) {
      if (!proto[ITERATOR]) hide(proto, ITERATOR, ArrayValues);
      if (!proto[TO_STRING_TAG]) hide(proto, TO_STRING_TAG, NAME);
      Iterators[NAME] = ArrayValues;
      if (explicit) for (key in $iterators) if (!proto[key]) redefine(proto, key, $iterators[key], true);
    }
  }


  /***/ }),

  /***/ "vhPU":
  /***/ (function(module, exports) {

  // 7.2.1 RequireObjectCoercible(argument)
  module.exports = function (it) {
    if (it == undefined) throw TypeError("Can't call method on  " + it);
    return it;
  };


  /***/ }),

  /***/ "w2a5":
  /***/ (function(module, exports, __webpack_require__) {

  // false -> Array#indexOf
  // true  -> Array#includes
  var toIObject = __webpack_require__("aCFj");
  var toLength = __webpack_require__("ne8i");
  var toAbsoluteIndex = __webpack_require__("d/Gc");
  module.exports = function (IS_INCLUDES) {
    return function ($this, el, fromIndex) {
      var O = toIObject($this);
      var length = toLength(O.length);
      var index = toAbsoluteIndex(fromIndex, length);
      var value;
      // Array#includes uses SameValueZero equality algorithm
      // eslint-disable-next-line no-self-compare
      if (IS_INCLUDES && el != el) while (length > index) {
        value = O[index++];
        // eslint-disable-next-line no-self-compare
        if (value != value) return true;
      // Array#indexOf ignores holes, Array#includes - not
      } else for (;length > index; index++) if (IS_INCLUDES || index in O) {
        if (O[index] === el) return IS_INCLUDES || index || 0;
      } return !IS_INCLUDES && -1;
    };
  };


  /***/ }),

  /***/ "xpql":
  /***/ (function(module, exports, __webpack_require__) {

  module.exports = !__webpack_require__("nh4g") && !__webpack_require__("eeVq")(function () {
    return Object.defineProperty(__webpack_require__("Iw71")('div'), 'a', { get: function () { return 7; } }).a != 7;
  });


  /***/ }),

  /***/ "y3w9":
  /***/ (function(module, exports, __webpack_require__) {

  var isObject = __webpack_require__("0/R4");
  module.exports = function (it) {
    if (!isObject(it)) throw TypeError(it + ' is not an object!');
    return it;
  };


  /***/ }),

  /***/ "ylqs":
  /***/ (function(module, exports) {

  var id = 0;
  var px = Math.random();
  module.exports = function (key) {
    return 'Symbol('.concat(key === undefined ? '' : key, ')_', (++id + px).toString(36));
  };


  /***/ }),

  /***/ "yt8O":
  /***/ (function(module, exports, __webpack_require__) {

  var addToUnscopables = __webpack_require__("nGyu");
  var step = __webpack_require__("1TsA");
  var Iterators = __webpack_require__("hPIQ");
  var toIObject = __webpack_require__("aCFj");

  // 22.1.3.4 Array.prototype.entries()
  // 22.1.3.13 Array.prototype.keys()
  // 22.1.3.29 Array.prototype.values()
  // 22.1.3.30 Array.prototype[@@iterator]()
  module.exports = __webpack_require__("Afnz")(Array, 'Array', function (iterated, kind) {
    this._t = toIObject(iterated); // target
    this._i = 0;                   // next index
    this._k = kind;                // kind
  // 22.1.5.2.1 %ArrayIteratorPrototype%.next()
  }, function () {
    var O = this._t;
    var kind = this._k;
    var index = this._i++;
    if (!O || index >= O.length) {
      this._t = undefined;
      return step(1);
    }
    if (kind == 'keys') return step(0, index);
    if (kind == 'values') return step(0, O[index]);
    return step(0, [index, O[index]]);
  }, 'values');

  // argumentsList[@@iterator] is %ArrayProto_values% (9.4.4.6, 9.4.4.7)
  Iterators.Arguments = Iterators.Array;

  addToUnscopables('keys');
  addToUnscopables('values');
  addToUnscopables('entries');


  /***/ }),

  /***/ "zhAb":
  /***/ (function(module, exports, __webpack_require__) {

  var has = __webpack_require__("aagx");
  var toIObject = __webpack_require__("aCFj");
  var arrayIndexOf = __webpack_require__("w2a5")(false);
  var IE_PROTO = __webpack_require__("YTvA")('IE_PROTO');

  module.exports = function (object, names) {
    var O = toIObject(object);
    var i = 0;
    var result = [];
    var key;
    for (key in O) if (key != IE_PROTO) has(O, key) && result.push(key);
    // Don't enum bug & hidden keys
    while (names.length > i) if (has(O, key = names[i++])) {
      ~arrayIndexOf(result, key) || result.push(key);
    }
    return result;
  };


  /***/ })

  /******/ })["default"];
  });

  });

  var DialogDrag = unwrapExports(vueDialogDrag_umd);

  //
  //
  //
  //
  //
  //

  var script = {
    name: 'HollowDotsSpinner',

    props: {
      animationDuration: {
        type: Number,
        default: 1000
      },
      dotSize: {
        type: Number,
        default: 15
      },
      dotsNum: {
        type: Number,
        default: 3
      },
      color: {
        type: String,
        default: '#fff'
      }
    },

    computed: {
      horizontalMargin () {
        return this.dotSize / 2
      },

      spinnerStyle () {
        return {
          height: `${this.dotSize}px`,
          width: `${(this.dotSize + this.horizontalMargin * 2) * this.dotsNum}px`
        }
      },

      dotStyle () {
        return {
          animationDuration: `${this.animationDuration}ms`,
          width: `${this.dotSize}px`,
          height: `${this.dotSize}px`,
          margin: `0 ${this.horizontalMargin}px`,
          borderWidth: `${this.dotSize / 5}px`,
          borderColor: this.color
        }
      },

      dotsStyles () {
        const dotsStyles = [];
        const delayModifier = 0.3;
        const basicDelay = 1000;

        for (let i = 1; i <= this.dotsNum; i++) {
          const style = Object.assign({
            animationDelay: `${basicDelay * i * delayModifier}ms`
          }, this.dotStyle);

          dotsStyles.push(style);
        }

        return dotsStyles
      }
    }
  };

  function normalizeComponent(template, style, script, scopeId, isFunctionalTemplate, moduleIdentifier
  /* server only */
  , shadowMode, createInjector, createInjectorSSR, createInjectorShadow) {
    if (typeof shadowMode !== 'boolean') {
      createInjectorSSR = createInjector;
      createInjector = shadowMode;
      shadowMode = false;
    } // Vue.extend constructor export interop.


    var options = typeof script === 'function' ? script.options : script; // render functions

    if (template && template.render) {
      options.render = template.render;
      options.staticRenderFns = template.staticRenderFns;
      options._compiled = true; // functional template

      if (isFunctionalTemplate) {
        options.functional = true;
      }
    } // scopedId


    if (scopeId) {
      options._scopeId = scopeId;
    }

    var hook;

    if (moduleIdentifier) {
      // server build
      hook = function hook(context) {
        // 2.3 injection
        context = context || // cached call
        this.$vnode && this.$vnode.ssrContext || // stateful
        this.parent && this.parent.$vnode && this.parent.$vnode.ssrContext; // functional
        // 2.2 with runInNewContext: true

        if (!context && typeof __VUE_SSR_CONTEXT__ !== 'undefined') {
          context = __VUE_SSR_CONTEXT__;
        } // inject component styles


        if (style) {
          style.call(this, createInjectorSSR(context));
        } // register component module identifier for async chunk inference


        if (context && context._registeredComponents) {
          context._registeredComponents.add(moduleIdentifier);
        }
      }; // used by ssr in case component is cached and beforeCreate
      // never gets called


      options._ssrRegister = hook;
    } else if (style) {
      hook = shadowMode ? function () {
        style.call(this, createInjectorShadow(this.$root.$options.shadowRoot));
      } : function (context) {
        style.call(this, createInjector(context));
      };
    }

    if (hook) {
      if (options.functional) {
        // register for functional component in vue file
        var originalRender = options.render;

        options.render = function renderWithStyleInjection(h, context) {
          hook.call(context);
          return originalRender(h, context);
        };
      } else {
        // inject component registration as beforeCreate hook
        var existing = options.beforeCreate;
        options.beforeCreate = existing ? [].concat(existing, hook) : [hook];
      }
    }

    return script;
  }

  var normalizeComponent_1 = normalizeComponent;

  var isOldIE = typeof navigator !== 'undefined' && /msie [6-9]\\b/.test(navigator.userAgent.toLowerCase());
  function createInjector(context) {
    return function (id, style) {
      return addStyle(id, style);
    };
  }
  var HEAD = document.head || document.getElementsByTagName('head')[0];
  var styles = {};

  function addStyle(id, css) {
    var group = isOldIE ? css.media || 'default' : id;
    var style = styles[group] || (styles[group] = {
      ids: new Set(),
      styles: []
    });

    if (!style.ids.has(id)) {
      style.ids.add(id);
      var code = css.source;

      if (css.map) {
        // https://developer.chrome.com/devtools/docs/javascript-debugging
        // this makes source maps inside style tags work properly in Chrome
        code += '\n/*# sourceURL=' + css.map.sources[0] + ' */'; // http://stackoverflow.com/a/26603875

        code += '\n/*# sourceMappingURL=data:application/json;base64,' + btoa(unescape(encodeURIComponent(JSON.stringify(css.map)))) + ' */';
      }

      if (!style.element) {
        style.element = document.createElement('style');
        style.element.type = 'text/css';
        if (css.media) style.element.setAttribute('media', css.media);
        HEAD.appendChild(style.element);
      }

      if ('styleSheet' in style.element) {
        style.styles.push(code);
        style.element.styleSheet.cssText = style.styles.filter(Boolean).join('\n');
      } else {
        var index = style.ids.size - 1;
        var textNode = document.createTextNode(code);
        var nodes = style.element.childNodes;
        if (nodes[index]) style.element.removeChild(nodes[index]);
        if (nodes.length) style.element.insertBefore(textNode, nodes[index]);else style.element.appendChild(textNode);
      }
    }
  }

  var browser = createInjector;

  /* script */
  const __vue_script__ = script;

  /* template */
  var __vue_render__ = function () {var _vm=this;var _h=_vm.$createElement;var _c=_vm._self._c||_h;return _c('div',{staticClass:"hollow-dots-spinner",style:(_vm.spinnerStyle)},_vm._l((_vm.dotsStyles),function(ds,index){return _c('div',{key:index,staticClass:"dot",style:(ds)})}),0)};
  var __vue_staticRenderFns__ = [];

    /* style */
    const __vue_inject_styles__ = function (inject) {
      if (!inject) return
      inject("data-v-25cabae1_0", { source: ".hollow-dots-spinner[data-v-25cabae1],.hollow-dots-spinner *[data-v-25cabae1]{box-sizing:border-box}.hollow-dots-spinner[data-v-25cabae1]{height:15px;width:calc(30px * 3)}.hollow-dots-spinner .dot[data-v-25cabae1]{width:15px;height:15px;margin:0 calc(15px / 2);border:calc(15px / 5) solid #ff1d5e;border-radius:50%;float:left;transform:scale(0);animation:hollow-dots-spinner-animation-data-v-25cabae1 1s ease infinite 0s}.hollow-dots-spinner .dot[data-v-25cabae1]:nth-child(1){animation-delay:calc(300ms * 1)}.hollow-dots-spinner .dot[data-v-25cabae1]:nth-child(2){animation-delay:calc(300ms * 2)}.hollow-dots-spinner .dot[data-v-25cabae1]:nth-child(3){animation-delay:calc(300ms * 3)}@keyframes hollow-dots-spinner-animation-data-v-25cabae1{50%{transform:scale(1);opacity:1}100%{opacity:0}}", map: undefined, media: undefined });

    };
    /* scoped */
    const __vue_scope_id__ = "data-v-25cabae1";
    /* module identifier */
    const __vue_module_identifier__ = undefined;
    /* functional template */
    const __vue_is_functional_template__ = false;
    /* style inject SSR */
    

    
    normalizeComponent_1(
      { render: __vue_render__, staticRenderFns: __vue_staticRenderFns__ },
      __vue_inject_styles__,
      __vue_script__,
      __vue_scope_id__,
      __vue_is_functional_template__,
      __vue_module_identifier__,
      browser,
      undefined
    );

  var utils$2 = {
    /**
     * Add reyframes to body as style block
     * @param name string
     * @param frames string
     */
    appendKeyframes: function (name, frames) {
      const sheet = document.createElement('style');
      if (!sheet) {
        return
      }
      sheet.setAttribute('id', name);
      sheet.innerHTML = `@keyframes ${name} {${frames}}`;
      document.body.appendChild(sheet);
    },
    /**
     * Remove reyframes from body
     * @param name string
     */
    removeKeyframes: function (name) {
      const sheet = document.getElementById(name);
      if (!sheet) {
        return
      }
      const sheetParent = sheet.parentNode;
      sheetParent.removeChild(sheet);
    }
  };

  //

  var script$1 = {
    name: 'PixelSpinner',

    props: {
      animationDuration: {
        type: Number,
        default: 1500
      },
      size: {
        type: Number,
        default: 70
      },
      color: {
        type: String,
        default: '#fff'
      }
    },

    data () {
      return {
        animationName: `pixel-spinner-animation-${Date.now()}`
      }
    },

    computed: {
      pixelSize () {
        return this.size / 7
      },

      spinnerStyle () {
        return {
          width: `${this.size}px`,
          height: `${this.size}px`
        }
      },

      spinnerInnerStyle () {
        return {
          animationDuration: `${this.animationDuration}ms`,
          animationName: this.animationName,
          width: `${this.pixelSize}px`,
          height: `${this.pixelSize}px`,
          backgroundColor: this.color,
          color: this.color,
          boxShadow: `
                      ${this.pixelSize * 1.5}px ${this.pixelSize * 1.5}px 0 0,
                      ${this.pixelSize * -1.5}px ${this.pixelSize * -1.5}px 0 0,
                      ${this.pixelSize * 1.5}px ${this.pixelSize * -1.5}px 0 0,
                      ${this.pixelSize * -1.5}px ${this.pixelSize * 1.5}px 0 0,
                      0 ${this.pixelSize * 1.5}px 0 0,
                      ${this.pixelSize * 1.5}px 0 0 0,
                      ${this.pixelSize * -1.5}px 0 0 0,
                      0 ${this.pixelSize * -1.5}px 0 0
                    `
        }
      }
    },

    watch: {
      size: {
        handler: 'updateAnimation',
        immediate: true
      }
    },

    mounted () {
      this.updateAnimation();
    },

    beforeDestroy () {
      utils$2.removeKeyframes(this.animationName);
    },

    methods: {
      updateAnimation () {
        utils$2.removeKeyframes(this.animationName);
        utils$2.appendKeyframes(this.animationName, this.generateKeyframes());
      },

      generateKeyframes () {
        return `
      50% {
        box-shadow:  ${this.pixelSize * 2}px ${this.pixelSize * 2}px 0 0,
                     ${this.pixelSize * -2}px ${this.pixelSize * -2}px 0 0,
                     ${this.pixelSize * 2}px ${this.pixelSize * -2}px 0 0,
                     ${this.pixelSize * -2}px ${this.pixelSize * 2}px 0 0,
                     0 ${this.pixelSize}px 0 0,
                     ${this.pixelSize}px 0 0 0,
                     ${this.pixelSize * -1}px 0 0 0,
                     0 ${this.pixelSize * -1}px 0 0;
      }


      75% {
        box-shadow:  ${this.pixelSize * 2}px ${this.pixelSize * 2}px 0 0,
                     ${this.pixelSize * -2}px ${this.pixelSize * -2}px 0 0,
                     ${this.pixelSize * 2}px ${this.pixelSize * -2}px 0 0,
                     ${this.pixelSize * -2}px ${this.pixelSize * 2}px 0 0,
                     0 ${this.pixelSize}px 0 0,
                     ${this.pixelSize}px 0 0 0,
                     ${this.pixelSize * -1}px 0 0 0,
                     0 ${this.pixelSize * -1}px 0 0;
      }

      100% {
        transform: rotate(360deg);
      }`
      }
    }
  };

  /* script */
  const __vue_script__$1 = script$1;

  /* template */
  var __vue_render__$1 = function () {var _vm=this;var _h=_vm.$createElement;var _c=_vm._self._c||_h;return _c('div',{staticClass:"pixel-spinner",style:(_vm.spinnerStyle)},[_c('div',{staticClass:"pixel-spinner-inner",style:(_vm.spinnerInnerStyle)})])};
  var __vue_staticRenderFns__$1 = [];

    /* style */
    const __vue_inject_styles__$1 = function (inject) {
      if (!inject) return
      inject("data-v-2eb24194_0", { source: ".pixel-spinner[data-v-2eb24194],.pixel-spinner *[data-v-2eb24194]{box-sizing:border-box}.pixel-spinner[data-v-2eb24194]{height:70px;width:70px;display:flex;flex-direction:row;justify-content:center;align-items:center}.pixel-spinner .pixel-spinner-inner[data-v-2eb24194]{width:calc(70px / 7);height:calc(70px / 7);background-color:#ff1d5e;color:#ff1d5e;box-shadow:15px 15px 0 0,-15px -15px 0 0,15px -15px 0 0,-15px 15px 0 0,0 15px 0 0,15px 0 0 0,-15px 0 0 0,0 -15px 0 0;animation:pixel-spinner-animation-data-v-2eb24194 2s linear infinite}@keyframes pixel-spinner-animation-data-v-2eb24194{50%{box-shadow:20px 20px 0 0,-20px -20px 0 0,20px -20px 0 0,-20px 20px 0 0,0 10px 0 0,10px 0 0 0,-10px 0 0 0,0 -10px 0 0}75%{box-shadow:20px 20px 0 0,-20px -20px 0 0,20px -20px 0 0,-20px 20px 0 0,0 10px 0 0,10px 0 0 0,-10px 0 0 0,0 -10px 0 0}100%{transform:rotate(360deg)}}", map: undefined, media: undefined });

    };
    /* scoped */
    const __vue_scope_id__$1 = "data-v-2eb24194";
    /* module identifier */
    const __vue_module_identifier__$1 = undefined;
    /* functional template */
    const __vue_is_functional_template__$1 = false;
    /* style inject SSR */
    

    
    normalizeComponent_1(
      { render: __vue_render__$1, staticRenderFns: __vue_staticRenderFns__$1 },
      __vue_inject_styles__$1,
      __vue_script__$1,
      __vue_scope_id__$1,
      __vue_is_functional_template__$1,
      __vue_module_identifier__$1,
      browser,
      undefined
    );

  //

  var script$2 = {
    name: 'FlowerSpinner',

    props: {
      animationDuration: {
        type: Number,
        default: 2500
      },
      size: {
        type: Number,
        default: 70
      },
      color: {
        type: String,
        default: '#fff'
      }
    },

    data () {
      return {
        smallDotName: `flower-spinner-small-dot-${Date.now()}`,
        bigDotName: `flower-spinner-big-dot-${Date.now()}`
      }
    },

    computed: {
      dotSize () {
        return this.size / 7
      },

      spinnerStyle () {
        return {
          width: `${this.size}px`,
          height: `${this.size}px`
        }
      },

      dotsContainerStyle () {
        return {
          width: `${this.dotSize}px`,
          height: `${this.dotSize}px`
        }
      },

      smallerDotStyle () {
        return {
          background: this.color,
          animationDuration: `${this.animationDuration}ms`,
          animationName: this.smallDotName
        }
      },

      biggerDotStyle () {
        return {
          background: this.color,
          animationDuration: `${this.animationDuration}ms`,
          animationName: this.bigDotName
        }
      }
    },

    watch: {
      size: {
        handler: 'updateAnimation',
        immediate: true
      },
      color: {
        handler: 'updateAnimation',
        immediate: true
      }
    },

    beforeDestroy () {
      utils$2.removeKeyframes(this.smallDotName);
      utils$2.removeKeyframes(this.bigDotName);
    },

    methods: {
      updateAnimation () {
        utils$2.removeKeyframes(this.smallDotName);
        utils$2.appendKeyframes(this.smallDotName, this.generateSmallDotKeyframes());
        utils$2.removeKeyframes(this.bigDotName);
        utils$2.appendKeyframes(this.bigDotName, this.generateBigDotKeyframes());
      },

      generateSmallDotKeyframes () {
        return `0%, 100% {
                  box-shadow: 0 0 0 ${this.color},
                              0 0 0 ${this.color},
                              0 0 0 ${this.color},
                              0 0 0 ${this.color},
                              0 0 0 ${this.color},
                              0 0 0 ${this.color},
                              0 0 0 ${this.color},
                              0 0 0 ${this.color};
                }
                25%, 75% {
                  box-shadow: ${this.dotSize * 1.4}px 0 0 ${this.color},
                              -${this.dotSize * 1.4}px 0 0 ${this.color},
                              0 ${this.dotSize * 1.4}px 0 ${this.color},
                              0 -${this.dotSize * 1.4}px 0 ${this.color},
                              ${this.dotSize}px -${this.dotSize}px 0 ${this.color},
                              ${this.dotSize}px ${this.dotSize}px 0 ${this.color},
                              -${this.dotSize}px -${this.dotSize}px 0 ${this.color},
                              -${this.dotSize}px ${this.dotSize}px 0 ${this.color};

                }
                100% {
                  box-shadow: 0 0 0 ${this.color},
                              0 0 0 ${this.color},
                              0 0 0 ${this.color},
                              0 0 0 ${this.color},
                              0 0 0 ${this.color},
                              0 0 0 ${this.color},
                              0 0 0 ${this.color},
                              0 0 0 ${this.color};
                }`
      },

      generateBigDotKeyframes () {
        return `0%, 100% {
                  box-shadow: 0 0 0 ${this.color},
                              0 0 0 ${this.color},
                              0 0 0 ${this.color},
                              0 0 0 ${this.color},
                              0 0 0 ${this.color},
                              0 0 0 ${this.color},
                              0 0 0 ${this.color},
                              0 0 0 ${this.color};
                }
                50% {
                  transform: rotate(180deg);
                }
                25%, 75% {
                  box-shadow: ${this.dotSize * 2.6}px 0 0 ${this.color},
                              -${this.dotSize * 2.6}px 0 0 ${this.color},
                              0 ${this.dotSize * 2.6}px 0 ${this.color},
                              0 -${this.dotSize * 2.6}px 0 ${this.color},
                              ${this.dotSize * 1.9}px -${this.dotSize * 1.9}px 0 ${this.color},
                              ${this.dotSize * 1.9}px ${this.dotSize * 1.9}px 0 ${this.color},
                              -${this.dotSize * 1.9}px -${this.dotSize * 1.9}px 0 ${this.color},
                              -${this.dotSize * 1.9}px ${this.dotSize * 1.9}px 0 ${this.color};

                }
                100% {
                  transform: rotate(360deg);
                  box-shadow: 0 0 0 ${this.color},
                              0 0 0 ${this.color},
                              0 0 0 ${this.color},
                              0 0 0 ${this.color},
                              0 0 0 ${this.color},
                              0 0 0 ${this.color},
                              0 0 0 ${this.color},
                              0 0 0 ${this.color};
                }`
      }
    }
  };

  /* script */
  const __vue_script__$2 = script$2;

  /* template */
  var __vue_render__$2 = function () {var _vm=this;var _h=_vm.$createElement;var _c=_vm._self._c||_h;return _c('div',{staticClass:"flower-spinner",style:(_vm.spinnerStyle)},[_c('div',{staticClass:"dots-container",style:(_vm.dotsContainerStyle)},[_c('div',{staticClass:"big-dot",style:(_vm.biggerDotStyle)},[_c('div',{staticClass:"small-dot",style:(_vm.smallerDotStyle)})])])])};
  var __vue_staticRenderFns__$2 = [];

    /* style */
    const __vue_inject_styles__$2 = function (inject) {
      if (!inject) return
      inject("data-v-53551658_0", { source: ".flower-spinner[data-v-53551658],.flower-spinner *[data-v-53551658]{box-sizing:border-box}.flower-spinner[data-v-53551658]{height:70px;width:70px;display:flex;flex-direction:row;align-items:center;justify-content:center}.flower-spinner .dots-container[data-v-53551658]{height:calc(70px / 7);width:calc(70px / 7)}.flower-spinner .small-dot[data-v-53551658]{background:#ff1d5e;height:100%;width:100%;border-radius:50%;animation:flower-spinner-small-dot-animation-data-v-53551658 2.5s 0s infinite both}.flower-spinner .big-dot[data-v-53551658]{background:#ff1d5e;height:100%;width:100%;padding:10%;border-radius:50%;animation:flower-spinner-big-dot-animation-data-v-53551658 2.5s 0s infinite both}@keyframes flower-spinner-big-dot-animation-data-v-53551658{0%,100%{box-shadow:#ff1d5e 0 0 0,#ff1d5e 0 0 0,#ff1d5e 0 0 0,#ff1d5e 0 0 0,#ff1d5e 0 0 0,#ff1d5e 0 0 0,#ff1d5e 0 0 0,#ff1d5e 0 0 0}50%{transform:rotate(180deg)}25%,75%{box-shadow:#ff1d5e 26px 0 0,#ff1d5e -26px 0 0,#ff1d5e 0 26px 0,#ff1d5e 0 -26px 0,#ff1d5e 19px -19px 0,#ff1d5e 19px 19px 0,#ff1d5e -19px -19px 0,#ff1d5e -19px 19px 0}100%{transform:rotate(360deg);box-shadow:#ff1d5e 0 0 0,#ff1d5e 0 0 0,#ff1d5e 0 0 0,#ff1d5e 0 0 0,#ff1d5e 0 0 0,#ff1d5e 0 0 0,#ff1d5e 0 0 0,#ff1d5e 0 0 0}}@keyframes flower-spinner-small-dot-animation-data-v-53551658{0%,100%{box-shadow:#ff1d5e 0 0 0,#ff1d5e 0 0 0,#ff1d5e 0 0 0,#ff1d5e 0 0 0,#ff1d5e 0 0 0,#ff1d5e 0 0 0,#ff1d5e 0 0 0,#ff1d5e 0 0 0}25%,75%{box-shadow:#ff1d5e 14px 0 0,#ff1d5e -14px 0 0,#ff1d5e 0 14px 0,#ff1d5e 0 -14px 0,#ff1d5e 10px -10px 0,#ff1d5e 10px 10px 0,#ff1d5e -10px -10px 0,#ff1d5e -10px 10px 0}100%{box-shadow:#ff1d5e 0 0 0,#ff1d5e 0 0 0,#ff1d5e 0 0 0,#ff1d5e 0 0 0,#ff1d5e 0 0 0,#ff1d5e 0 0 0,#ff1d5e 0 0 0,#ff1d5e 0 0 0}}", map: undefined, media: undefined });

    };
    /* scoped */
    const __vue_scope_id__$2 = "data-v-53551658";
    /* module identifier */
    const __vue_module_identifier__$2 = undefined;
    /* functional template */
    const __vue_is_functional_template__$2 = false;
    /* style inject SSR */
    

    
    normalizeComponent_1(
      { render: __vue_render__$2, staticRenderFns: __vue_staticRenderFns__$2 },
      __vue_inject_styles__$2,
      __vue_script__$2,
      __vue_scope_id__$2,
      __vue_is_functional_template__$2,
      __vue_module_identifier__$2,
      browser,
      undefined
    );

  //
  //
  //
  //
  //
  //
  //
  //

  var script$3 = {
    name: 'IntersectingCirclesSpinner',

    props: {
      animationDuration: {
        type: Number,
        default: 1200
      },
      size: {
        type: Number,
        default: 70
      },
      color: {
        type: String,
        default: '#fff'
      }
    },

    computed: {
      circleSize () {
        return this.size / 2
      },

      spinnerStyle () {
        return {
          width: `${this.size}px`,
          height: `${this.size}px`
        }
      },

      spinnerBlockStyle () {
        return {
          animationDuration: `${this.animationDuration}ms`,
          width: `${this.circleSize}px`,
          height: `${this.circleSize}px`
        }
      },

      circleStyle () {
        return {
          borderColor: this.color
        }
      },

      circleStyles () {
        const circlesPositions = [
          {top: 0, left: 0},
          {left: `${this.circleSize * -0.36}px`, top: `${this.circleSize * 0.2}px`},
          {left: `${this.circleSize * -0.36}px`, top: `${this.circleSize * -0.2}px`},
          {left: 0, top: `${this.circleSize * -0.36}px`},
          {left: `${this.circleSize * 0.36}px`, top: `${this.circleSize * -0.2}px`},
          {left: `${this.circleSize * 0.36}px`, top: `${this.circleSize * 0.2}px`},
          {left: 0, top: `${this.circleSize * 0.36}px`}
        ];

        return circlesPositions.map((cp) => Object.assign(cp, this.circleStyle))
      }
    }
  };

  /* script */
  const __vue_script__$3 = script$3;

  /* template */
  var __vue_render__$3 = function () {var _vm=this;var _h=_vm.$createElement;var _c=_vm._self._c||_h;return _c('div',{staticClass:"intersecting-circles-spinner",style:(_vm.spinnerStyle)},[_c('div',{staticClass:"spinnerBlock",style:(_vm.spinnerBlockStyle)},_vm._l((_vm.circleStyles),function(cs,index){return _c('span',{key:index,staticClass:"circle",style:(cs)})}),0)])};
  var __vue_staticRenderFns__$3 = [];

    /* style */
    const __vue_inject_styles__$3 = function (inject) {
      if (!inject) return
      inject("data-v-41c291c2_0", { source: ".intersecting-circles-spinner[data-v-41c291c2],.intersecting-circles-spinner *[data-v-41c291c2]{box-sizing:border-box}.intersecting-circles-spinner[data-v-41c291c2]{height:70px;width:70px;position:relative;display:flex;flex-direction:row;justify-content:center;align-items:center}.intersecting-circles-spinner .spinnerBlock[data-v-41c291c2]{animation:intersecting-circles-spinners-animation-data-v-41c291c2 1.2s linear infinite;transform-origin:center;display:block;height:35px;width:35px}.intersecting-circles-spinner .circle[data-v-41c291c2]{display:block;border:2px solid #ff1d5e;border-radius:50%;height:100%;width:100%;position:absolute;left:0;top:0}.intersecting-circles-spinner .circle[data-v-41c291c2]:nth-child(1){left:0;top:0}.intersecting-circles-spinner .circle[data-v-41c291c2]:nth-child(2){left:calc(35px * -.36);top:calc(35px * .2)}.intersecting-circles-spinner .circle[data-v-41c291c2]:nth-child(3){left:calc(35px * -.36);top:calc(35px * -.2)}.intersecting-circles-spinner .circle[data-v-41c291c2]:nth-child(4){left:0;top:calc(35px * -.36)}.intersecting-circles-spinner .circle[data-v-41c291c2]:nth-child(5){left:calc(35px * .36);top:calc(35px * -.2)}.intersecting-circles-spinner .circle[data-v-41c291c2]:nth-child(6){left:calc(35px * .36);top:calc(35px * .2)}.intersecting-circles-spinner .circle[data-v-41c291c2]:nth-child(7){left:0;top:calc(35px * .36)}@keyframes intersecting-circles-spinners-animation-data-v-41c291c2{from{transform:rotate(0)}to{transform:rotate(360deg)}}", map: undefined, media: undefined });

    };
    /* scoped */
    const __vue_scope_id__$3 = "data-v-41c291c2";
    /* module identifier */
    const __vue_module_identifier__$3 = undefined;
    /* functional template */
    const __vue_is_functional_template__$3 = false;
    /* style inject SSR */
    

    
    normalizeComponent_1(
      { render: __vue_render__$3, staticRenderFns: __vue_staticRenderFns__$3 },
      __vue_inject_styles__$3,
      __vue_script__$3,
      __vue_scope_id__$3,
      __vue_is_functional_template__$3,
      __vue_module_identifier__$3,
      browser,
      undefined
    );

  //
  //
  //
  //
  //
  //
  //
  //

  var script$4 = {
    name: 'OrbitSpinner',

    props: {
      animationDuration: {
        type: Number,
        default: 1000
      },
      size: {
        type: Number,
        default: 50
      },
      color: {
        type: String,
        default: '#fff'
      }
    },

    computed: {
      spinnerStyle () {
        return {
          height: `${this.size}px`,
          width: `${this.size}px`
        }
      },

      orbitStyle () {
        return {
          borderColor: this.color,
          animationDuration: `${this.animationDuration}ms`
        }
      }
    }
  };

  /* script */
  const __vue_script__$4 = script$4;

  /* template */
  var __vue_render__$4 = function () {var _vm=this;var _h=_vm.$createElement;var _c=_vm._self._c||_h;return _c('div',{staticClass:"orbit-spinner",style:(_vm.spinnerStyle)},[_c('div',{staticClass:"orbit one",style:(_vm.orbitStyle)}),_vm._v(" "),_c('div',{staticClass:"orbit two",style:(_vm.orbitStyle)}),_vm._v(" "),_c('div',{staticClass:"orbit three",style:(_vm.orbitStyle)})])};
  var __vue_staticRenderFns__$4 = [];

    /* style */
    const __vue_inject_styles__$4 = function (inject) {
      if (!inject) return
      inject("data-v-57121865_0", { source: ".orbit-spinner[data-v-57121865],.orbit-spinner *[data-v-57121865]{box-sizing:border-box}.orbit-spinner[data-v-57121865]{height:55px;width:55px;border-radius:50%;perspective:800px}.orbit-spinner .orbit[data-v-57121865]{position:absolute;box-sizing:border-box;width:100%;height:100%;border-radius:50%}.orbit-spinner .orbit[data-v-57121865]:nth-child(1){left:0;top:0;animation:orbit-spinner-orbit-one-animation-data-v-57121865 1.2s linear infinite;border-bottom:3px solid #ff1d5e}.orbit-spinner .orbit[data-v-57121865]:nth-child(2){right:0;top:0;animation:orbit-spinner-orbit-two-animation-data-v-57121865 1.2s linear infinite;border-right:3px solid #ff1d5e}.orbit-spinner .orbit[data-v-57121865]:nth-child(3){right:0;bottom:0;animation:orbit-spinner-orbit-three-animation-data-v-57121865 1.2s linear infinite;border-top:3px solid #ff1d5e}@keyframes orbit-spinner-orbit-one-animation-data-v-57121865{0%{transform:rotateX(35deg) rotateY(-45deg) rotateZ(0)}100%{transform:rotateX(35deg) rotateY(-45deg) rotateZ(360deg)}}@keyframes orbit-spinner-orbit-two-animation-data-v-57121865{0%{transform:rotateX(50deg) rotateY(10deg) rotateZ(0)}100%{transform:rotateX(50deg) rotateY(10deg) rotateZ(360deg)}}@keyframes orbit-spinner-orbit-three-animation-data-v-57121865{0%{transform:rotateX(35deg) rotateY(55deg) rotateZ(0)}100%{transform:rotateX(35deg) rotateY(55deg) rotateZ(360deg)}}", map: undefined, media: undefined });

    };
    /* scoped */
    const __vue_scope_id__$4 = "data-v-57121865";
    /* module identifier */
    const __vue_module_identifier__$4 = undefined;
    /* functional template */
    const __vue_is_functional_template__$4 = false;
    /* style inject SSR */
    

    
    normalizeComponent_1(
      { render: __vue_render__$4, staticRenderFns: __vue_staticRenderFns__$4 },
      __vue_inject_styles__$4,
      __vue_script__$4,
      __vue_scope_id__$4,
      __vue_is_functional_template__$4,
      __vue_module_identifier__$4,
      browser,
      undefined
    );

  //
  //
  //
  //
  //
  //

  var script$5 = {
    name: 'FingerprintSpinner',

    props: {
      animationDuration: {
        type: Number,
        default: 1500
      },
      size: {
        type: Number,
        default: 60
      },
      color: {
        type: String,
        default: '#fff'
      }
    },

    data () {
      return {
        ringsNum: 9,
        containerPadding: 2
      }
    },

    computed: {
      outerRingSize () {
        return this.size - this.containerPadding * 2
      },

      spinnerStyle () {
        return {
          height: `${this.size}px`,
          width: `${this.size}px`,
          padding: `${this.containerPadding}px`
        }
      },

      ringStyle () {
        return {
          borderTopColor: this.color,
          animationDuration: `${this.animationDuration}ms`
        }
      },

      ringsStyles () {
        const ringsStyles = [];
        const ringBase = this.outerRingSize / (this.ringsNum);
        const ringInc = ringBase;

        for (let i = 1; i <= this.ringsNum; i++) {
          let style = Object.assign({
            animationDelay: `${i * 50}ms`,
            height: `${ringBase + (i - 1) * ringInc}px`,
            width: `${ringBase + (i - 1) * ringInc}px`
          }, this.ringStyle);
          ringsStyles.push(style);
        }

        return ringsStyles
      }
    }
  };

  /* script */
  const __vue_script__$5 = script$5;

  /* template */
  var __vue_render__$5 = function () {var _vm=this;var _h=_vm.$createElement;var _c=_vm._self._c||_h;return _c('div',{staticClass:"fingerprint-spinner",style:(_vm.spinnerStyle)},_vm._l((_vm.ringsStyles),function(rs,index){return _c('div',{key:index,staticClass:"spinner-ring",style:(rs)})}),0)};
  var __vue_staticRenderFns__$5 = [];

    /* style */
    const __vue_inject_styles__$5 = function (inject) {
      if (!inject) return
      inject("data-v-398a6b6b_0", { source: ".fingerprint-spinner[data-v-398a6b6b],.fingerprint-spinner *[data-v-398a6b6b]{box-sizing:border-box}.fingerprint-spinner[data-v-398a6b6b]{height:64px;width:64px;padding:2px;overflow:hidden;position:relative}.fingerprint-spinner .spinner-ring[data-v-398a6b6b]{position:absolute;border-radius:50%;border:2px solid transparent;border-top-color:#ff1d5e;animation:fingerprint-spinner-animation-data-v-398a6b6b 1.5s cubic-bezier(.68,-.75,.265,1.75) infinite forwards;margin:auto;bottom:0;left:0;right:0;top:0}.fingerprint-spinner .spinner-ring[data-v-398a6b6b]:nth-child(1){height:calc(60px / 9 + 0 * 60px / 9);width:calc(60px / 9 + 0 * 60px / 9);animation-delay:calc(50ms * 1)}.fingerprint-spinner .spinner-ring[data-v-398a6b6b]:nth-child(2){height:calc(60px / 9 + 1 * 60px / 9);width:calc(60px / 9 + 1 * 60px / 9);animation-delay:calc(50ms * 2)}.fingerprint-spinner .spinner-ring[data-v-398a6b6b]:nth-child(3){height:calc(60px / 9 + 2 * 60px / 9);width:calc(60px / 9 + 2 * 60px / 9);animation-delay:calc(50ms * 3)}.fingerprint-spinner .spinner-ring[data-v-398a6b6b]:nth-child(4){height:calc(60px / 9 + 3 * 60px / 9);width:calc(60px / 9 + 3 * 60px / 9);animation-delay:calc(50ms * 4)}.fingerprint-spinner .spinner-ring[data-v-398a6b6b]:nth-child(5){height:calc(60px / 9 + 4 * 60px / 9);width:calc(60px / 9 + 4 * 60px / 9);animation-delay:calc(50ms * 5)}.fingerprint-spinner .spinner-ring[data-v-398a6b6b]:nth-child(6){height:calc(60px / 9 + 5 * 60px / 9);width:calc(60px / 9 + 5 * 60px / 9);animation-delay:calc(50ms * 6)}.fingerprint-spinner .spinner-ring[data-v-398a6b6b]:nth-child(7){height:calc(60px / 9 + 6 * 60px / 9);width:calc(60px / 9 + 6 * 60px / 9);animation-delay:calc(50ms * 7)}.fingerprint-spinner .spinner-ring[data-v-398a6b6b]:nth-child(8){height:calc(60px / 9 + 7 * 60px / 9);width:calc(60px / 9 + 7 * 60px / 9);animation-delay:calc(50ms * 8)}.fingerprint-spinner .spinner-ring[data-v-398a6b6b]:nth-child(9){height:calc(60px / 9 + 8 * 60px / 9);width:calc(60px / 9 + 8 * 60px / 9);animation-delay:calc(50ms * 9)}@keyframes fingerprint-spinner-animation-data-v-398a6b6b{100%{transform:rotate(360deg)}}", map: undefined, media: undefined });

    };
    /* scoped */
    const __vue_scope_id__$5 = "data-v-398a6b6b";
    /* module identifier */
    const __vue_module_identifier__$5 = undefined;
    /* functional template */
    const __vue_is_functional_template__$5 = false;
    /* style inject SSR */
    

    
    normalizeComponent_1(
      { render: __vue_render__$5, staticRenderFns: __vue_staticRenderFns__$5 },
      __vue_inject_styles__$5,
      __vue_script__$5,
      __vue_scope_id__$5,
      __vue_is_functional_template__$5,
      __vue_module_identifier__$5,
      browser,
      undefined
    );

  //
  //
  //
  //
  //
  //
  //
  //

  var script$6 = {
    name: 'TrinityRingsSpinner',

    props: {
      animationDuration: {
        type: Number,
        default: 1500
      },
      size: {
        type: Number,
        default: 60
      },
      color: {
        type: String,
        default: '#fff'
      }
    },

    data () {
      return {
        containerPadding: 3
      }
    },

    computed: {
      outerRingSize () {
        return this.size - this.containerPadding * 2
      },

      spinnerStyle () {
        return {
          height: `${this.size}px`,
          width: `${this.size}px`,
          padding: `${this.containerPadding}px`
        }
      },

      ring1Style () {
        return {
          height: `${this.outerRingSize}px`,
          width: `${this.outerRingSize}px`,
          borderColor: this.color,
          animationDuration: `${this.animationDuration}ms`
        }
      },

      ring2Style () {
        return {
          height: `${this.outerRingSize * 0.65}px`,
          width: `${this.outerRingSize * 0.65}px`,
          borderColor: this.color,
          animationDuration: `${this.animationDuration}ms`
        }
      },

      ring3Style () {
        return {
          height: `${this.outerRingSize * 0.1}px`,
          width: `${this.outerRingSize * 0.1}px`,
          borderColor: this.color,
          animationDuration: `${this.animationDuration}ms`
        }
      }
    }
  };

  /* script */
  const __vue_script__$6 = script$6;

  /* template */
  var __vue_render__$6 = function () {var _vm=this;var _h=_vm.$createElement;var _c=_vm._self._c||_h;return _c('div',{staticClass:"trinity-rings-spinner",style:(_vm.spinnerStyle)},[_c('div',{staticClass:"circle circle1",style:(_vm.ring1Style)}),_vm._v(" "),_c('div',{staticClass:"circle circle2",style:(_vm.ring2Style)}),_vm._v(" "),_c('div',{staticClass:"circle circle3",style:(_vm.ring3Style)})])};
  var __vue_staticRenderFns__$6 = [];

    /* style */
    const __vue_inject_styles__$6 = function (inject) {
      if (!inject) return
      inject("data-v-f744cb78_0", { source: ".trinity-rings-spinner[data-v-f744cb78],.trinity-rings-spinner *[data-v-f744cb78]{box-sizing:border-box}.trinity-rings-spinner[data-v-f744cb78]{height:66px;width:66px;padding:3px;position:relative;display:flex;justify-content:center;align-items:center;flex-direction:row;overflow:hidden;box-sizing:border-box}.trinity-rings-spinner .circle[data-v-f744cb78]{position:absolute;display:block;border-radius:50%;border:3px solid #ff1d5e;opacity:1}.trinity-rings-spinner .circle[data-v-f744cb78]:nth-child(1){height:60px;width:60px;animation:trinity-rings-spinner-circle1-animation-data-v-f744cb78 1.5s infinite linear;border-width:3px}.trinity-rings-spinner .circle[data-v-f744cb78]:nth-child(2){height:calc(60px * .65);width:calc(60px * .65);animation:trinity-rings-spinner-circle2-animation-data-v-f744cb78 1.5s infinite linear;border-width:2px}.trinity-rings-spinner .circle[data-v-f744cb78]:nth-child(3){height:calc(60px * .1);width:calc(60px * .1);animation:trinity-rings-spinner-circle3-animation-data-v-f744cb78 1.5s infinite linear;border-width:1px}@keyframes trinity-rings-spinner-circle1-animation-data-v-f744cb78{0%{transform:rotateZ(20deg) rotateY(0)}100%{transform:rotateZ(100deg) rotateY(360deg)}}@keyframes trinity-rings-spinner-circle2-animation-data-v-f744cb78{0%{transform:rotateZ(100deg) rotateX(0)}100%{transform:rotateZ(0) rotateX(360deg)}}@keyframes trinity-rings-spinner-circle3-animation-data-v-f744cb78{0%{transform:rotateZ(100deg) rotateX(-360deg)}100%{transform:rotateZ(-360deg) rotateX(360deg)}}", map: undefined, media: undefined });

    };
    /* scoped */
    const __vue_scope_id__$6 = "data-v-f744cb78";
    /* module identifier */
    const __vue_module_identifier__$6 = undefined;
    /* functional template */
    const __vue_is_functional_template__$6 = false;
    /* style inject SSR */
    

    
    normalizeComponent_1(
      { render: __vue_render__$6, staticRenderFns: __vue_staticRenderFns__$6 },
      __vue_inject_styles__$6,
      __vue_script__$6,
      __vue_scope_id__$6,
      __vue_is_functional_template__$6,
      __vue_module_identifier__$6,
      browser,
      undefined
    );

  //
  //
  //
  //
  //
  //

  var script$7 = {
    name: 'FulfillingSquareSpinner',

    props: {
      animationDuration: {
        type: Number,
        default: 4000
      },
      size: {
        type: Number,
        default: 50
      },
      color: {
        type: String,
        default: '#fff'
      }
    },

    computed: {
      spinnerStyle () {
        return {
          height: `${this.size}px`,
          width: `${this.size}px`,
          borderColor: this.color
        }
      },

      spinnerInnerStyle () {
        return {
          backgroundColor: this.color
        }
      }
    }
  };

  /* script */
  const __vue_script__$7 = script$7;

  /* template */
  var __vue_render__$7 = function () {var _vm=this;var _h=_vm.$createElement;var _c=_vm._self._c||_h;return _c('div',{staticClass:"fulfilling-square-spinner",style:(_vm.spinnerStyle)},[_c('div',{staticClass:"spinner-inner",style:(_vm.spinnerInnerStyle)})])};
  var __vue_staticRenderFns__$7 = [];

    /* style */
    const __vue_inject_styles__$7 = function (inject) {
      if (!inject) return
      inject("data-v-05b759a7_0", { source: ".fulfilling-square-spinner[data-v-05b759a7],.fulfilling-square-spinner *[data-v-05b759a7]{box-sizing:border-box}.fulfilling-square-spinner[data-v-05b759a7]{height:50px;width:50px;position:relative;border:4px solid #ff1d5e;animation:fulfilling-square-spinner-animation-data-v-05b759a7 4s infinite ease}.fulfilling-square-spinner .spinner-inner[data-v-05b759a7]{vertical-align:top;display:inline-block;background-color:#ff1d5e;width:100%;opacity:1;animation:fulfilling-square-spinner-inner-animation-data-v-05b759a7 4s infinite ease-in}@keyframes fulfilling-square-spinner-animation-data-v-05b759a7{0%{transform:rotate(0)}25%{transform:rotate(180deg)}50%{transform:rotate(180deg)}75%{transform:rotate(360deg)}100%{transform:rotate(360deg)}}@keyframes fulfilling-square-spinner-inner-animation-data-v-05b759a7{0%{height:0%}25%{height:0%}50%{height:100%}75%{height:100%}100%{height:0%}}", map: undefined, media: undefined });

    };
    /* scoped */
    const __vue_scope_id__$7 = "data-v-05b759a7";
    /* module identifier */
    const __vue_module_identifier__$7 = undefined;
    /* functional template */
    const __vue_is_functional_template__$7 = false;
    /* style inject SSR */
    

    
    normalizeComponent_1(
      { render: __vue_render__$7, staticRenderFns: __vue_staticRenderFns__$7 },
      __vue_inject_styles__$7,
      __vue_script__$7,
      __vue_scope_id__$7,
      __vue_is_functional_template__$7,
      __vue_module_identifier__$7,
      browser,
      undefined
    );

  //
  //
  //
  //
  //
  //

  var script$8 = {
    name: 'CirclesToRhombusesSpinner',

    props: {
      animationDuration: {
        type: Number,
        default: 1200
      },
      circleSize: {
        type: Number,
        default: 15
      },
      color: {
        type: String,
        default: '#fff'
      },
      circlesNum: {
        type: Number,
        default: 3
      }
    },

    computed: {
      circleMarginLeft () {
        return this.circleSize * 1.125
      },

      spinnertStyle () {
        return {
          height: `${this.circleSize}px`,
          width: `${(this.circleSize + this.circleMarginLeft) * this.circlesNum}px`
        }
      },

      circleStyle () {
        return {
          borderColor: this.color,
          animationDuration: `${this.animationDuration}ms`,
          height: `${this.circleSize}px`,
          width: `${this.circleSize}px`,
          marginLeft: `${this.circleMarginLeft}px`
        }
      },

      circlesStyles () {
        const circlesStyles = [];
        const delay = 150;

        for (let i = 1; i <= this.circlesNum; i++) {
          const style = Object.assign({
            animationDelay: `${i * delay}ms`
          }, this.circleStyle);

          if (i === 1) {
            style.marginLeft = 0;
          }

          circlesStyles.push(style);
        }

        return circlesStyles
      }
    }
  };

  /* script */
  const __vue_script__$8 = script$8;

  /* template */
  var __vue_render__$8 = function () {var _vm=this;var _h=_vm.$createElement;var _c=_vm._self._c||_h;return _c('div',{staticClass:"circles-to-rhombuses-spinner",style:(_vm.spinnertStyle)},_vm._l((_vm.circlesStyles),function(cs,index){return _c('div',{key:index,staticClass:"circle",style:(cs)})}),0)};
  var __vue_staticRenderFns__$8 = [];

    /* style */
    const __vue_inject_styles__$8 = function (inject) {
      if (!inject) return
      inject("data-v-19f0105a_0", { source: ".circles-to-rhombuses-spinner[data-v-19f0105a],.circles-to-rhombuses-spinner *[data-v-19f0105a]{box-sizing:border-box}.circles-to-rhombuses-spinner[data-v-19f0105a]{height:15px;width:calc((15px + 15px * 1.125) * 3);display:flex;align-items:center;justify-content:center}.circles-to-rhombuses-spinner .circle[data-v-19f0105a]{height:15px;width:15px;margin-left:calc(15px * 1.125);transform:rotate(45deg);border-radius:10%;border:3px solid #ff1d5e;overflow:hidden;background:0 0;animation:circles-to-rhombuses-animation-data-v-19f0105a 1.2s linear infinite}.circles-to-rhombuses-spinner .circle[data-v-19f0105a]:nth-child(1){animation-delay:calc(150ms * 1);margin-left:0}.circles-to-rhombuses-spinner .circle[data-v-19f0105a]:nth-child(2){animation-delay:calc(150ms * 2)}.circles-to-rhombuses-spinner .circle[data-v-19f0105a]:nth-child(3){animation-delay:calc(150ms * 3)}@keyframes circles-to-rhombuses-animation-data-v-19f0105a{0%{border-radius:10%}17.5%{border-radius:10%}50%{border-radius:100%}93.5%{border-radius:10%}100%{border-radius:10%}}@keyframes circles-to-rhombuses-background-animation-data-v-19f0105a{50%{opacity:.4}}", map: undefined, media: undefined });

    };
    /* scoped */
    const __vue_scope_id__$8 = "data-v-19f0105a";
    /* module identifier */
    const __vue_module_identifier__$8 = undefined;
    /* functional template */
    const __vue_is_functional_template__$8 = false;
    /* style inject SSR */
    

    
    normalizeComponent_1(
      { render: __vue_render__$8, staticRenderFns: __vue_staticRenderFns__$8 },
      __vue_inject_styles__$8,
      __vue_script__$8,
      __vue_scope_id__$8,
      __vue_is_functional_template__$8,
      __vue_module_identifier__$8,
      browser,
      undefined
    );

  //
  //
  //
  //
  //
  //

  var script$9 = {
    name: 'SemipolarSpinner',

    props: {
      animationDuration: {
        type: Number,
        default: 2000
      },
      size: {
        type: Number,
        default: 65
      },
      color: {
        type: String,
        default: '#fff'
      }
    },

    data () {
      return {
        ringsNum: 5
      }
    },

    computed: {
      spinnerStyle () {
        return {
          height: `${this.size}px`,
          width: `${this.size}px`
        }
      },
      ringStyle () {
        return {
          animationDuration: `${this.animationDuration}ms`,
          borderTopColor: this.color,
          borderLeftColor: this.color
        }
      },
      ringsStyles () {
        const ringsStyles = [];
        const delayModifier = 0.1;
        const ringWidth = this.size * 0.05;
        const positionIncrement = ringWidth * 2;
        const sizeDecrement = this.size * 0.2;

        for (let i = 0; i < this.ringsNum; i++) {
          const computedSize = `${this.size - sizeDecrement * i}px`;
          const computedPosition = `${positionIncrement * i}px`;
          const style = Object.assign({
            animationDelay: `${this.animationDuration * delayModifier * (this.ringsNum - i - 1)}ms`,
            height: computedSize,
            width: computedSize,
            left: computedPosition,
            top: computedPosition,
            borderWidth: `${ringWidth}px`
          }, this.ringStyle);
          ringsStyles.push(style);
        }

        return ringsStyles
      }
    }
  };

  /* script */
  const __vue_script__$9 = script$9;

  /* template */
  var __vue_render__$9 = function () {var _vm=this;var _h=_vm.$createElement;var _c=_vm._self._c||_h;return _c('div',{staticClass:"semipolar-spinner",style:(_vm.spinnerStyle)},_vm._l((_vm.ringsStyles),function(rs,index){return _c('div',{key:index,staticClass:"ring",style:(rs)})}),0)};
  var __vue_staticRenderFns__$9 = [];

    /* style */
    const __vue_inject_styles__$9 = function (inject) {
      if (!inject) return
      inject("data-v-3e6fcf9d_0", { source: ".semipolar-spinner[data-v-3e6fcf9d],.semipolar-spinner *[data-v-3e6fcf9d]{box-sizing:border-box}.semipolar-spinner[data-v-3e6fcf9d]{height:65px;width:65px;position:relative}.semipolar-spinner .ring[data-v-3e6fcf9d]{border-radius:50%;position:absolute;border:calc(65px * .05) solid transparent;border-top-color:#ff1d5e;border-left-color:#ff1d5e;animation:semipolar-spinner-animation-data-v-3e6fcf9d 2s infinite}.semipolar-spinner .ring[data-v-3e6fcf9d]:nth-child(1){height:calc(65px - 65px * .2 * 0);width:calc(65px - 65px * .2 * 0);top:calc(65px * .1 * 0);left:calc(65px * .1 * 0);animation-delay:calc(2000ms * .1 * 4);z-index:5}.semipolar-spinner .ring[data-v-3e6fcf9d]:nth-child(2){height:calc(65px - 65px * .2 * 1);width:calc(65px - 65px * .2 * 1);top:calc(65px * .1 * 1);left:calc(65px * .1 * 1);animation-delay:calc(2000ms * .1 * 3);z-index:4}.semipolar-spinner .ring[data-v-3e6fcf9d]:nth-child(3){height:calc(65px - 65px * .2 * 2);width:calc(65px - 65px * .2 * 2);top:calc(65px * .1 * 2);left:calc(65px * .1 * 2);animation-delay:calc(2000ms * .1 * 2);z-index:3}.semipolar-spinner .ring[data-v-3e6fcf9d]:nth-child(4){height:calc(65px - 65px * .2 * 3);width:calc(65px - 65px * .2 * 3);top:calc(65px * .1 * 3);left:calc(65px * .1 * 3);animation-delay:calc(2000ms * .1 * 1);z-index:2}.semipolar-spinner .ring[data-v-3e6fcf9d]:nth-child(5){height:calc(65px - 65px * .2 * 4);width:calc(65px - 65px * .2 * 4);top:calc(65px * .1 * 4);left:calc(65px * .1 * 4);animation-delay:calc(2000ms * .1 * 0);z-index:1}@keyframes semipolar-spinner-animation-data-v-3e6fcf9d{50%{transform:rotate(360deg) scale(.7)}}", map: undefined, media: undefined });

    };
    /* scoped */
    const __vue_scope_id__$9 = "data-v-3e6fcf9d";
    /* module identifier */
    const __vue_module_identifier__$9 = undefined;
    /* functional template */
    const __vue_is_functional_template__$9 = false;
    /* style inject SSR */
    

    
    normalizeComponent_1(
      { render: __vue_render__$9, staticRenderFns: __vue_staticRenderFns__$9 },
      __vue_inject_styles__$9,
      __vue_script__$9,
      __vue_scope_id__$9,
      __vue_is_functional_template__$9,
      __vue_module_identifier__$9,
      browser,
      undefined
    );

  //
  //
  //
  //
  //
  //
  //

  var script$a = {
    name: 'BreedingRhombusSpinner',

    props: {
      animationDuration: {
        type: Number,
        default: 2000
      },
      size: {
        type: Number,
        default: 150
      },
      color: {
        type: String,
        default: '#fff'
      }
    },

    data () {
      return {
        animationBaseName: 'breeding-rhombus-spinner-animation-child',
        rhombusesNum: 8
      }
    },

    computed: {
      spinnerStyle () {
        return {
          height: `${this.size}px`,
          width: `${this.size}px`
        }
      },

      rhombusStyle () {
        return {
          height: `${this.size / 7.5}px`,
          width: `${this.size / 7.5}px`,
          animationDuration: `${this.animationDuration}ms`,
          top: `${this.size / 2.3077}px`,
          left: `${this.size / 2.3077}px`,
          backgroundColor: this.color
        }
      },

      rhombusesStyles () {
        const rhombusesStyles = [];
        const delayModifier = this.animationDuration * 0.05;

        for (let i = 1; i <= this.rhombusesNum; i++) {
          rhombusesStyles.push(Object.assign({
            animationDelay: `${delayModifier * (i + 1)}ms`
          }, this.rhombusStyle));
        }

        return rhombusesStyles
      },

      bigRhombusStyle () {
        return {
          height: `${this.size / 3}px`,
          width: `${this.size / 3}px`,
          animationDuration: `${this.animationDuration}`,
          top: `${this.size / 3}px`,
          left: `${this.size / 3}px`,
          backgroundColor: this.color
        }
      }
    }

  };

  /* script */
  const __vue_script__$a = script$a;

  /* template */
  var __vue_render__$a = function () {var _vm=this;var _h=_vm.$createElement;var _c=_vm._self._c||_h;return _c('div',{staticClass:"breeding-rhombus-spinner",style:(_vm.spinnerStyle)},[_vm._l((_vm.rhombusesStyles),function(rs,index){return _c('div',{key:index,staticClass:"rhombus",class:("child-" + (index + 1)),style:(rs)})}),_vm._v(" "),_c('div',{staticClass:"rhombus big",style:(_vm.bigRhombusStyle)})],2)};
  var __vue_staticRenderFns__$a = [];

    /* style */
    const __vue_inject_styles__$a = function (inject) {
      if (!inject) return
      inject("data-v-363a4360_0", { source: ".breeding-rhombus-spinner[data-v-363a4360]{height:65px;width:65px;position:relative;transform:rotate(45deg)}.breeding-rhombus-spinner[data-v-363a4360],.breeding-rhombus-spinner *[data-v-363a4360]{box-sizing:border-box}.breeding-rhombus-spinner .rhombus[data-v-363a4360]{height:calc(65px / 7.5);width:calc(65px / 7.5);animation-duration:2s;top:calc(65px / 2.3077);left:calc(65px / 2.3077);background-color:#ff1d5e;position:absolute;animation-iteration-count:infinite}.breeding-rhombus-spinner .rhombus[data-v-363a4360]:nth-child(2n+0){margin-right:0}.breeding-rhombus-spinner .rhombus.child-1[data-v-363a4360]{animation-name:breeding-rhombus-spinner-animation-child-1-data-v-363a4360;animation-delay:calc(100ms * 1)}.breeding-rhombus-spinner .rhombus.child-2[data-v-363a4360]{animation-name:breeding-rhombus-spinner-animation-child-2-data-v-363a4360;animation-delay:calc(100ms * 2)}.breeding-rhombus-spinner .rhombus.child-3[data-v-363a4360]{animation-name:breeding-rhombus-spinner-animation-child-3-data-v-363a4360;animation-delay:calc(100ms * 3)}.breeding-rhombus-spinner .rhombus.child-4[data-v-363a4360]{animation-name:breeding-rhombus-spinner-animation-child-4-data-v-363a4360;animation-delay:calc(100ms * 4)}.breeding-rhombus-spinner .rhombus.child-5[data-v-363a4360]{animation-name:breeding-rhombus-spinner-animation-child-5-data-v-363a4360;animation-delay:calc(100ms * 5)}.breeding-rhombus-spinner .rhombus.child-6[data-v-363a4360]{animation-name:breeding-rhombus-spinner-animation-child-6-data-v-363a4360;animation-delay:calc(100ms * 6)}.breeding-rhombus-spinner .rhombus.child-7[data-v-363a4360]{animation-name:breeding-rhombus-spinner-animation-child-7-data-v-363a4360;animation-delay:calc(100ms * 7)}.breeding-rhombus-spinner .rhombus.child-8[data-v-363a4360]{animation-name:breeding-rhombus-spinner-animation-child-8-data-v-363a4360;animation-delay:calc(100ms * 8)}.breeding-rhombus-spinner .rhombus.big[data-v-363a4360]{height:calc(65px / 3);width:calc(65px / 3);animation-duration:2s;top:calc(65px / 3);left:calc(65px / 3);background-color:#ff1d5e;animation:breeding-rhombus-spinner-animation-child-big-data-v-363a4360 2s infinite;animation-delay:.5s}@keyframes breeding-rhombus-spinner-animation-child-1-data-v-363a4360{50%{transform:translate(-325%,-325%)}}@keyframes breeding-rhombus-spinner-animation-child-2-data-v-363a4360{50%{transform:translate(0,-325%)}}@keyframes breeding-rhombus-spinner-animation-child-3-data-v-363a4360{50%{transform:translate(325%,-325%)}}@keyframes breeding-rhombus-spinner-animation-child-4-data-v-363a4360{50%{transform:translate(325%,0)}}@keyframes breeding-rhombus-spinner-animation-child-5-data-v-363a4360{50%{transform:translate(325%,325%)}}@keyframes breeding-rhombus-spinner-animation-child-6-data-v-363a4360{50%{transform:translate(0,325%)}}@keyframes breeding-rhombus-spinner-animation-child-7-data-v-363a4360{50%{transform:translate(-325%,325%)}}@keyframes breeding-rhombus-spinner-animation-child-8-data-v-363a4360{50%{transform:translate(-325%,0)}}@keyframes breeding-rhombus-spinner-animation-child-big-data-v-363a4360{50%{transform:scale(.5)}}", map: undefined, media: undefined });

    };
    /* scoped */
    const __vue_scope_id__$a = "data-v-363a4360";
    /* module identifier */
    const __vue_module_identifier__$a = undefined;
    /* functional template */
    const __vue_is_functional_template__$a = false;
    /* style inject SSR */
    

    
    normalizeComponent_1(
      { render: __vue_render__$a, staticRenderFns: __vue_staticRenderFns__$a },
      __vue_inject_styles__$a,
      __vue_script__$a,
      __vue_scope_id__$a,
      __vue_is_functional_template__$a,
      __vue_module_identifier__$a,
      browser,
      undefined
    );

  //
  //
  //
  //
  //
  //
  //
  //
  //
  //
  //

  var script$b = {
    name: 'SwappingSquaresSpinner',

    props: {
      animationDuration: {
        type: Number,
        default: 1000
      },
      size: {
        type: Number,
        default: 65
      },
      color: {
        type: String,
        default: '#fff'
      }
    },

    data () {
      return {
        animationBaseName: 'swapping-squares-animation-child',
        squaresNum: 4
      }
    },

    computed: {
      spinnerStyle () {
        return {
          height: `${this.size}px`,
          width: `${this.size}px`
        }
      },

      squareStyle () {
        return {
          height: `${this.size * 0.25 / 1.3}px`,
          width: `${this.size * 0.25 / 1.3}px`,
          animationDuration: `${this.animationDuration}ms`,
          borderWidth: `${this.size * 0.04 / 1.3}px`,
          borderColor: this.color
        }
      },

      squaresStyles () {
        const squaresStyles = [];
        const delay = this.animationDuration * 0.5;

        for (let i = 1; i <= this.squaresNum; i++) {
          squaresStyles.push(Object.assign({
            animationDelay: `${i % 2 === 0 ? delay : 0}ms`
          }, this.squareStyle));
        }

        return squaresStyles
      }
    }
  };

  /* script */
  const __vue_script__$b = script$b;

  /* template */
  var __vue_render__$b = function () {var _vm=this;var _h=_vm.$createElement;var _c=_vm._self._c||_h;return _c('div',{staticClass:"swapping-squares-spinner",style:(_vm.spinnerStyle)},_vm._l((_vm.squaresStyles),function(ss,index){return _c('div',{key:index,staticClass:"square",class:("square-" + (index + 1)),style:(ss)})}),0)};
  var __vue_staticRenderFns__$b = [];

    /* style */
    const __vue_inject_styles__$b = function (inject) {
      if (!inject) return
      inject("data-v-03072400_0", { source: ".swapping-squares-spinner[data-v-03072400],.swapping-squares-spinner *[data-v-03072400]{box-sizing:border-box}.swapping-squares-spinner[data-v-03072400]{height:65px;width:65px;position:relative;display:flex;flex-direction:row;justify-content:center;align-items:center}.swapping-squares-spinner .square[data-v-03072400]{height:calc(65px * .25 / 1.3);width:calc(65px * .25 / 1.3);animation-duration:1s;border:calc(65px * .04 / 1.3) solid #ff1d5e;margin-right:auto;margin-left:auto;position:absolute;animation-iteration-count:infinite}.swapping-squares-spinner .square[data-v-03072400]:nth-child(1){animation-name:swapping-squares-animation-child-1-data-v-03072400;animation-delay:.5s}.swapping-squares-spinner .square[data-v-03072400]:nth-child(2){animation-name:swapping-squares-animation-child-2-data-v-03072400;animation-delay:0s}.swapping-squares-spinner .square[data-v-03072400]:nth-child(3){animation-name:swapping-squares-animation-child-3-data-v-03072400;animation-delay:.5s}.swapping-squares-spinner .square[data-v-03072400]:nth-child(4){animation-name:swapping-squares-animation-child-4-data-v-03072400;animation-delay:0s}@keyframes swapping-squares-animation-child-1-data-v-03072400{50%{transform:translate(150%,150%) scale(2,2)}}@keyframes swapping-squares-animation-child-2-data-v-03072400{50%{transform:translate(-150%,150%) scale(2,2)}}@keyframes swapping-squares-animation-child-3-data-v-03072400{50%{transform:translate(-150%,-150%) scale(2,2)}}@keyframes swapping-squares-animation-child-4-data-v-03072400{50%{transform:translate(150%,-150%) scale(2,2)}}", map: undefined, media: undefined });

    };
    /* scoped */
    const __vue_scope_id__$b = "data-v-03072400";
    /* module identifier */
    const __vue_module_identifier__$b = undefined;
    /* functional template */
    const __vue_is_functional_template__$b = false;
    /* style inject SSR */
    

    
    normalizeComponent_1(
      { render: __vue_render__$b, staticRenderFns: __vue_staticRenderFns__$b },
      __vue_inject_styles__$b,
      __vue_script__$b,
      __vue_scope_id__$b,
      __vue_is_functional_template__$b,
      __vue_module_identifier__$b,
      browser,
      undefined
    );

  //
  //
  //
  //
  //
  //
  //
  //
  //
  //
  //

  var script$c = {
    name: 'ScalingSquaresSpinner',

    props: {
      animationDuration: {
        type: Number,
        default: 1250
      },
      size: {
        type: Number,
        default: 65
      },
      color: {
        type: String,
        default: '#fff'
      }
    },

    data () {
      return {
        squaresNum: 4
      }
    },

    computed: {
      spinnerStyle () {
        return {
          height: `${this.size}px`,
          width: `${this.size}px`,
          animationDuration: `${this.animationDuration}ms`
        }
      },

      squareStyle () {
        return {
          height: `${this.size * 0.25 / 1.3}px`,
          width: `${this.size * 0.25 / 1.3}px`,
          animationDuration: `${this.animationDuration}ms`,
          borderWidth: `${this.size * 0.04 / 1.3}px`,
          borderColor: this.color
        }
      },

      squaresStyles () {
        const squaresStyles = [];

        for (let i = 1; i <= this.squaresNum; i++) {
          squaresStyles.push(Object.assign({
          }, this.squareStyle));
        }

        return squaresStyles
      }
    }
  };

  /* script */
  const __vue_script__$c = script$c;

  /* template */
  var __vue_render__$c = function () {var _vm=this;var _h=_vm.$createElement;var _c=_vm._self._c||_h;return _c('div',{staticClass:"scaling-squares-spinner",style:(_vm.spinnerStyle)},_vm._l((_vm.squaresStyles),function(ss,index){return _c('div',{key:index,staticClass:"square",class:("square-" + (index + 1)),style:(ss)})}),0)};
  var __vue_staticRenderFns__$c = [];

    /* style */
    const __vue_inject_styles__$c = function (inject) {
      if (!inject) return
      inject("data-v-7d3af2b9_0", { source: ".scaling-squares-spinner[data-v-7d3af2b9],.scaling-squares-spinner *[data-v-7d3af2b9]{box-sizing:border-box}.scaling-squares-spinner[data-v-7d3af2b9]{height:65px;width:65px;position:relative;display:flex;flex-direction:row;align-items:center;justify-content:center;animation:scaling-squares-animation-data-v-7d3af2b9 1.25s;animation-iteration-count:infinite;transform:rotate(0)}.scaling-squares-spinner .square[data-v-7d3af2b9]{height:calc(65px * .25 / 1.3);width:calc(65px * .25 / 1.3);margin-right:auto;margin-left:auto;border:calc(65px * .04 / 1.3) solid #ff1d5e;position:absolute;animation-duration:1.25s;animation-iteration-count:infinite}.scaling-squares-spinner .square[data-v-7d3af2b9]:nth-child(1){animation-name:scaling-squares-spinner-animation-child-1-data-v-7d3af2b9}.scaling-squares-spinner .square[data-v-7d3af2b9]:nth-child(2){animation-name:scaling-squares-spinner-animation-child-2-data-v-7d3af2b9}.scaling-squares-spinner .square[data-v-7d3af2b9]:nth-child(3){animation-name:scaling-squares-spinner-animation-child-3-data-v-7d3af2b9}.scaling-squares-spinner .square[data-v-7d3af2b9]:nth-child(4){animation-name:scaling-squares-spinner-animation-child-4-data-v-7d3af2b9}@keyframes scaling-squares-animation-data-v-7d3af2b9{50%{transform:rotate(90deg)}100%{transform:rotate(180deg)}}@keyframes scaling-squares-spinner-animation-child-1-data-v-7d3af2b9{50%{transform:translate(150%,150%) scale(2,2)}}@keyframes scaling-squares-spinner-animation-child-2-data-v-7d3af2b9{50%{transform:translate(-150%,150%) scale(2,2)}}@keyframes scaling-squares-spinner-animation-child-3-data-v-7d3af2b9{50%{transform:translate(-150%,-150%) scale(2,2)}}@keyframes scaling-squares-spinner-animation-child-4-data-v-7d3af2b9{50%{transform:translate(150%,-150%) scale(2,2)}}", map: undefined, media: undefined });

    };
    /* scoped */
    const __vue_scope_id__$c = "data-v-7d3af2b9";
    /* module identifier */
    const __vue_module_identifier__$c = undefined;
    /* functional template */
    const __vue_is_functional_template__$c = false;
    /* style inject SSR */
    

    
    normalizeComponent_1(
      { render: __vue_render__$c, staticRenderFns: __vue_staticRenderFns__$c },
      __vue_inject_styles__$c,
      __vue_script__$c,
      __vue_scope_id__$c,
      __vue_is_functional_template__$c,
      __vue_module_identifier__$c,
      browser,
      undefined
    );

  //
  //
  //
  //
  //
  //
  //

  var script$d = {
    name: 'FulfillingBouncingCircleSpinner',

    props: {
      animationDuration: {
        type: Number,
        default: 4000
      },
      size: {
        type: Number,
        default: 60
      },
      color: {
        type: String,
        default: '#fff'
      }
    },

    computed: {
      spinnerStyle () {
        return {
          height: `${this.size}px`,
          width: `${this.size}px`,
          animationDuration: `${this.animationDuration}ms`
        }
      },

      orbitStyle () {
        return {
          height: `${this.size}px`,
          width: `${this.size}px`,
          borderColor: this.color,
          borderWidth: `${this.size * 0.03}px`,
          animationDuration: `${this.animationDuration}ms`
        }
      },

      circleStyle () {
        return {
          height: `${this.size}px`,
          width: `${this.size}px`,
          borderColor: this.color,
          color: this.color,
          borderWidth: `${this.size * 0.1}px`,
          animationDuration: `${this.animationDuration}ms`
        }
      }
    }
  };

  /* script */
  const __vue_script__$d = script$d;

  /* template */
  var __vue_render__$d = function () {var _vm=this;var _h=_vm.$createElement;var _c=_vm._self._c||_h;return _c('div',{staticClass:"fulfilling-bouncing-circle-spinner",style:(_vm.spinnerStyle)},[_c('div',{staticClass:"circle",style:(_vm.circleStyle)}),_vm._v(" "),_c('div',{staticClass:"orbit",style:(_vm.orbitStyle)})])};
  var __vue_staticRenderFns__$d = [];

    /* style */
    const __vue_inject_styles__$d = function (inject) {
      if (!inject) return
      inject("data-v-1d764dac_0", { source: ".fulfilling-bouncing-circle-spinner[data-v-1d764dac],.fulfilling-bouncing-circle-spinner *[data-v-1d764dac]{box-sizing:border-box}.fulfilling-bouncing-circle-spinner[data-v-1d764dac]{height:60px;width:60px;position:relative;animation:fulfilling-bouncing-circle-spinner-animation-data-v-1d764dac infinite 4s ease}.fulfilling-bouncing-circle-spinner .orbit[data-v-1d764dac]{height:60px;width:60px;position:absolute;top:0;left:0;border-radius:50%;border:calc(60px * .03) solid #ff1d5e;animation:fulfilling-bouncing-circle-spinner-orbit-animation-data-v-1d764dac infinite 4s ease}.fulfilling-bouncing-circle-spinner .circle[data-v-1d764dac]{height:60px;width:60px;color:#ff1d5e;display:block;border-radius:50%;position:relative;border:calc(60px * .1) solid #ff1d5e;animation:fulfilling-bouncing-circle-spinner-circle-animation-data-v-1d764dac infinite 4s ease;transform:rotate(0) scale(1)}@keyframes fulfilling-bouncing-circle-spinner-animation-data-v-1d764dac{0%{transform:rotate(0)}100%{transform:rotate(360deg)}}@keyframes fulfilling-bouncing-circle-spinner-orbit-animation-data-v-1d764dac{0%{transform:scale(1)}50%{transform:scale(1)}62.5%{transform:scale(.8)}75%{transform:scale(1)}87.5%{transform:scale(.8)}100%{transform:scale(1)}}@keyframes fulfilling-bouncing-circle-spinner-circle-animation-data-v-1d764dac{0%{transform:scale(1);border-color:transparent;border-top-color:inherit}16.7%{border-color:transparent;border-top-color:initial;border-right-color:initial}33.4%{border-color:transparent;border-top-color:inherit;border-right-color:inherit;border-bottom-color:inherit}50%{border-color:inherit;transform:scale(1)}62.5%{border-color:inherit;transform:scale(1.4)}75%{border-color:inherit;transform:scale(1);opacity:1}87.5%{border-color:inherit;transform:scale(1.4)}100%{border-color:transparent;border-top-color:inherit;transform:scale(1)}}", map: undefined, media: undefined });

    };
    /* scoped */
    const __vue_scope_id__$d = "data-v-1d764dac";
    /* module identifier */
    const __vue_module_identifier__$d = undefined;
    /* functional template */
    const __vue_is_functional_template__$d = false;
    /* style inject SSR */
    

    
    var FulfillingBouncingCircleSpinner = normalizeComponent_1(
      { render: __vue_render__$d, staticRenderFns: __vue_staticRenderFns__$d },
      __vue_inject_styles__$d,
      __vue_script__$d,
      __vue_scope_id__$d,
      __vue_is_functional_template__$d,
      __vue_module_identifier__$d,
      browser,
      undefined
    );

  //
  //
  //
  //
  //
  //
  //
  //
  //
  //

  var script$e = {
    name: 'RadarSpinner',

    props: {
      animationDuration: {
        type: Number,
        default: 2000
      },
      size: {
        type: Number,
        default: 110
      },
      color: {
        type: String,
        default: '#fff'
      }
    },

    data () {
      return {
        circlesNum: 4
      }
    },

    computed: {
      borderWidth () {
        return this.size * 5 / 110
      },

      spinnerStyle () {
        return {
          height: `${this.size}px`,
          width: `${this.size}px`
        }
      },

      circleStyle () {
        return {
          animationDuration: `${this.animationDuration}ms`
        }
      },

      circleInnerContainerStyle () {
        return {
          borderWidth: `${this.borderWidth}px`
        }
      },

      circleInnerStyle () {
        return {
          borderLeftColor: this.color,
          borderRightColor: this.color,
          borderWidth: `${this.borderWidth}px`
        }
      },

      circlesStyles () {
        const circlesStyles = [];
        const delay = this.animationDuration * 0.15;

        for (let i = 0; i < this.circlesNum; i++) {
          circlesStyles.push(Object.assign({
            padding: `${this.borderWidth * 2 * i}px`,
            animationDelay: `${i === this.circlesNum - 1 ? 0 : delay}ms`
          }, this.circleStyle));
        }

        return circlesStyles
      }
    }
  };

  /* script */
  const __vue_script__$e = script$e;

  /* template */
  var __vue_render__$e = function () {var _vm=this;var _h=_vm.$createElement;var _c=_vm._self._c||_h;return _c('div',{staticClass:"radar-spinner",style:(_vm.spinnerStyle)},_vm._l((_vm.circlesStyles),function(cs,index){return _c('div',{key:index,staticClass:"circle",style:(cs)},[_c('div',{staticClass:"circle-inner-container",style:(_vm.circleInnerContainerStyle)},[_c('div',{staticClass:"circle-inner",style:(_vm.circleInnerStyle)})])])}),0)};
  var __vue_staticRenderFns__$e = [];

    /* style */
    const __vue_inject_styles__$e = function (inject) {
      if (!inject) return
      inject("data-v-3817b858_0", { source: ".radar-spinner[data-v-3817b858],.radar-spinner *[data-v-3817b858]{box-sizing:border-box}.radar-spinner[data-v-3817b858]{height:60px;width:60px;position:relative}.radar-spinner .circle[data-v-3817b858]{position:absolute;height:100%;width:100%;top:0;left:0;animation:radar-spinner-animation-data-v-3817b858 2s infinite}.radar-spinner .circle[data-v-3817b858]:nth-child(1){padding:calc(60px * 5 * 2 * 0 / 110);animation-delay:.3s}.radar-spinner .circle[data-v-3817b858]:nth-child(2){padding:calc(60px * 5 * 2 * 1 / 110);animation-delay:.3s}.radar-spinner .circle[data-v-3817b858]:nth-child(3){padding:calc(60px * 5 * 2 * 2 / 110);animation-delay:.3s}.radar-spinner .circle[data-v-3817b858]:nth-child(4){padding:calc(60px * 5 * 2 * 3 / 110);animation-delay:0s}.radar-spinner .circle-inner[data-v-3817b858],.radar-spinner .circle-inner-container[data-v-3817b858]{height:100%;width:100%;border-radius:50%;border:calc(60px * 5 / 110) solid transparent}.radar-spinner .circle-inner[data-v-3817b858]{border-left-color:#ff1d5e;border-right-color:#ff1d5e}@keyframes radar-spinner-animation-data-v-3817b858{50%{transform:rotate(180deg)}100%{transform:rotate(0)}}", map: undefined, media: undefined });

    };
    /* scoped */
    const __vue_scope_id__$e = "data-v-3817b858";
    /* module identifier */
    const __vue_module_identifier__$e = undefined;
    /* functional template */
    const __vue_is_functional_template__$e = false;
    /* style inject SSR */
    

    
    normalizeComponent_1(
      { render: __vue_render__$e, staticRenderFns: __vue_staticRenderFns__$e },
      __vue_inject_styles__$e,
      __vue_script__$e,
      __vue_scope_id__$e,
      __vue_is_functional_template__$e,
      __vue_module_identifier__$e,
      browser,
      undefined
    );

  //
  //
  //
  //
  //
  //
  //
  //
  //
  //
  //
  //

  var script$f = {
    name: 'SelfBuildingSquareSpinner',

    props: {
      animationDuration: {
        type: Number,
        default: 6000
      },
      size: {
        type: Number,
        default: 40
      },
      color: {
        type: String,
        default: '#fff'
      }
    },

    data () {
      return {
        squaresNum: 9
      }
    },

    computed: {
      squareSize () {
        return this.size / 4
      },

      initialTopPosition () {
        return -this.squareSize * 2 / 3
      },

      spinnerStyle () {
        return {
          top: `${-this.initialTopPosition}px`,
          height: `${this.size}px`,
          width: `${this.size}px`
        }
      },

      squareStyle () {
        return {
          height: `${this.squareSize}px`,
          width: `${this.squareSize}px`,
          top: `${this.initialTopPosition}px`,
          marginRight: `${this.squareSize / 3}px`,
          marginTop: `${this.squareSize / 3}px`,
          animationDuration: `${this.animationDuration}ms`,
          background: this.color
        }
      },

      squaresStyles () {
        const squaresStyles = [];
        const delaysMultipliers = [6, 7, 8, 3, 4, 5, 0, 1, 2];
        const delayModifier = this.animationDuration * 0.05;

        for (let i = 0; i < this.squaresNum; i++) {
          squaresStyles.push(Object.assign({
            animationDelay: `${delayModifier * delaysMultipliers[i]}ms`
          }, this.squareStyle));
        }

        return squaresStyles
      }
    }
  };

  /* script */
  const __vue_script__$f = script$f;

  /* template */
  var __vue_render__$f = function () {var _vm=this;var _h=_vm.$createElement;var _c=_vm._self._c||_h;return _c('div',{staticClass:"self-building-square-spinner",style:(_vm.spinnerStyle)},_vm._l((_vm.squaresStyles),function(ss,index){return _c('div',{key:index,staticClass:"square",class:{'clear': index && index % 3 === 0},style:(ss)})}),0)};
  var __vue_staticRenderFns__$f = [];

    /* style */
    const __vue_inject_styles__$f = function (inject) {
      if (!inject) return
      inject("data-v-ee680776_0", { source: ".self-building-square-spinner[data-v-ee680776],.self-building-square-spinner *[data-v-ee680776]{box-sizing:border-box}.self-building-square-spinner[data-v-ee680776]{height:40px;width:40px;top:calc(-10px * 2 / 3)}.self-building-square-spinner .square[data-v-ee680776]{height:10px;width:10px;top:calc(-10px * 2 / 3);margin-right:calc(10px / 3);margin-top:calc(10px / 3);background:#ff1d5e;float:left;position:relative;opacity:0;animation:self-building-square-spinner-data-v-ee680776 6s infinite}.self-building-square-spinner .square[data-v-ee680776]:nth-child(1){animation-delay:calc(300ms * 6)}.self-building-square-spinner .square[data-v-ee680776]:nth-child(2){animation-delay:calc(300ms * 7)}.self-building-square-spinner .square[data-v-ee680776]:nth-child(3){animation-delay:calc(300ms * 8)}.self-building-square-spinner .square[data-v-ee680776]:nth-child(4){animation-delay:calc(300ms * 3)}.self-building-square-spinner .square[data-v-ee680776]:nth-child(5){animation-delay:calc(300ms * 4)}.self-building-square-spinner .square[data-v-ee680776]:nth-child(6){animation-delay:calc(300ms * 5)}.self-building-square-spinner .square[data-v-ee680776]:nth-child(7){animation-delay:calc(300ms * 0)}.self-building-square-spinner .square[data-v-ee680776]:nth-child(8){animation-delay:calc(300ms * 1)}.self-building-square-spinner .square[data-v-ee680776]:nth-child(9){animation-delay:calc(300ms * 2)}.self-building-square-spinner .clear[data-v-ee680776]{clear:both}@keyframes self-building-square-spinner-data-v-ee680776{0%{opacity:0}5%{opacity:1;top:0}50.9%{opacity:1;top:0}55.9%{opacity:0;top:inherit}}", map: undefined, media: undefined });

    };
    /* scoped */
    const __vue_scope_id__$f = "data-v-ee680776";
    /* module identifier */
    const __vue_module_identifier__$f = undefined;
    /* functional template */
    const __vue_is_functional_template__$f = false;
    /* style inject SSR */
    

    
    normalizeComponent_1(
      { render: __vue_render__$f, staticRenderFns: __vue_staticRenderFns__$f },
      __vue_inject_styles__$f,
      __vue_script__$f,
      __vue_scope_id__$f,
      __vue_is_functional_template__$f,
      __vue_module_identifier__$f,
      browser,
      undefined
    );

  //

  var script$g = {
    name: 'SpringSpinner',

    props: {
      animationDuration: {
        type: Number,
        default: 3000
      },
      size: {
        type: Number,
        default: 70
      },
      color: {
        type: String,
        default: '#fff'
      }
    },

    data () {
      return {
        animationName: `spring-spinner-animation-${Date.now()}`
      }
    },

    computed: {
      spinnerStyle () {
        return {
          height: `${this.size}px`,
          width: `${this.size}px`
        }
      },

      spinnerPartStyle () {
        return {
          height: `${this.size / 2}px`,
          width: `${this.size}px`
        }
      },

      rotatorStyle () {
        return {
          height: `${this.size}px`,
          width: `${this.size}px`,
          borderRightColor: this.color,
          borderTopColor: this.color,
          borderWidth: `${this.size / 7}px`,
          animationDuration: `${this.animationDuration}ms`,
          animationName: this.animationName
        }
      }
    },

    watch: {
      size: {
        handler: 'updateAnimation',
        immediate: true
      },
      color: {
        handler: 'updateAnimation',
        immediate: true
      }
    },

    mounted () {
      this.updateAnimation();
    },

    beforeDestroy () {
      utils$2.removeKeyframes(this.animationName);
    },

    methods: {
      updateAnimation () {
        utils$2.removeKeyframes(this.animationName);
        utils$2.appendKeyframes(this.animationName, this.generateKeyframes());
      },

      generateKeyframes () {
        return `
        0% {
          border-width: ${this.size / 7}px;
        }
        25% {
          border-width: ${this.size / 23.33}px;
        }
        50% {
          transform: rotate(115deg);
          border-width: ${this.size / 7}px;
        }
        75% {
          border-width: ${this.size / 23.33}px;
         }
        100% {
         border-width: ${this.size / 7}px;
        }`
      }
    }
  };

  /* script */
  const __vue_script__$g = script$g;

  /* template */
  var __vue_render__$g = function () {var _vm=this;var _h=_vm.$createElement;var _c=_vm._self._c||_h;return _c('div',{staticClass:"spring-spinner",style:(_vm.spinnerStyle)},[_c('div',{staticClass:"spring-spinner-part top",style:(_vm.spinnerPartStyle)},[_c('div',{staticClass:"spring-spinner-rotator",style:(_vm.rotatorStyle)})]),_vm._v(" "),_c('div',{staticClass:"spring-spinner-part bottom",style:(_vm.spinnerPartStyle)},[_c('div',{staticClass:"spring-spinner-rotator",style:(_vm.rotatorStyle)})])])};
  var __vue_staticRenderFns__$g = [];

    /* style */
    const __vue_inject_styles__$g = function (inject) {
      if (!inject) return
      inject("data-v-26c7dac8_0", { source: ".spring-spinner[data-v-26c7dac8],.spring-spinner *[data-v-26c7dac8]{box-sizing:border-box}.spring-spinner[data-v-26c7dac8]{height:60px;width:60px}.spring-spinner .spring-spinner-part[data-v-26c7dac8]{overflow:hidden;height:calc(60px / 2);width:60px}.spring-spinner .spring-spinner-part.bottom[data-v-26c7dac8]{transform:rotate(180deg) scale(-1,1)}.spring-spinner .spring-spinner-rotator[data-v-26c7dac8]{width:60px;height:60px;border:calc(60px / 7) solid transparent;border-right-color:#ff1d5e;border-top-color:#ff1d5e;border-radius:50%;box-sizing:border-box;animation:spring-spinner-animation-data-v-26c7dac8 3s ease-in-out infinite;transform:rotate(-200deg)}@keyframes spring-spinner-animation-data-v-26c7dac8{0%{border-width:calc(60px / 7)}25%{border-width:calc(60px / 23.33)}50%{transform:rotate(115deg);border-width:calc(60px / 7)}75%{border-width:calc(60px / 23.33)}100%{border-width:calc(60px / 7)}}", map: undefined, media: undefined });

    };
    /* scoped */
    const __vue_scope_id__$g = "data-v-26c7dac8";
    /* module identifier */
    const __vue_module_identifier__$g = undefined;
    /* functional template */
    const __vue_is_functional_template__$g = false;
    /* style inject SSR */
    

    
    normalizeComponent_1(
      { render: __vue_render__$g, staticRenderFns: __vue_staticRenderFns__$g },
      __vue_inject_styles__$g,
      __vue_script__$g,
      __vue_scope_id__$g,
      __vue_is_functional_template__$g,
      __vue_module_identifier__$g,
      browser,
      undefined
    );

  //
  //
  //
  //
  //
  //
  //
  //
  //
  //

  var script$h = {
    name: 'LoopingRhombusesSpinner',

    props: {
      animationDuration: {
        type: Number,
        default: 2500
      },
      rhombusSize: {
        type: Number,
        default: 15
      },
      color: {
        type: String,
        default: '#fff'
      }
    },

    data () {
      return {
        rhombusesNum: 3
      }
    },

    computed: {
      spinnerStyle () {
        return {
          height: `${this.rhombusSize}px`,
          width: `${this.rhombusSize * 4}px`
        }
      },

      rhombusStyle () {
        return {
          height: `${this.rhombusSize}px`,
          width: `${this.rhombusSize}px`,
          backgroundColor: this.color,
          animationDuration: `${this.animationDuration}ms`,
          left: `${this.rhombusSize * 4}px`
        }
      },

      rhombusesStyles () {
        const rhombusesStyles = [];
        const delay = -this.animationDuration / 1.5;

        for (let i = 1; i <= this.rhombusesNum; i++) {
          const style = Object.assign({
            animationDelay: `${i * delay}ms`
          }, this.rhombusStyle);

          rhombusesStyles.push(style);
        }

        return rhombusesStyles
      }
    }
  };

  /* script */
  const __vue_script__$h = script$h;

  /* template */
  var __vue_render__$h = function () {var _vm=this;var _h=_vm.$createElement;var _c=_vm._self._c||_h;return _c('div',{staticClass:"looping-rhombuses-spinner",style:(_vm.spinnerStyle)},_vm._l((_vm.rhombusesStyles),function(rs,index){return _c('div',{staticClass:"rhombus",style:(rs),attrs:{"ikey":index}})}),0)};
  var __vue_staticRenderFns__$h = [];

    /* style */
    const __vue_inject_styles__$h = function (inject) {
      if (!inject) return
      inject("data-v-e013f2d4_0", { source: ".looping-rhombuses-spinner[data-v-e013f2d4],.looping-rhombuses-spinner *[data-v-e013f2d4]{box-sizing:border-box}.looping-rhombuses-spinner[data-v-e013f2d4]{width:calc(15px * 4);height:15px;position:relative}.looping-rhombuses-spinner .rhombus[data-v-e013f2d4]{height:15px;width:15px;background-color:#ff1d5e;left:calc(15px * 4);position:absolute;margin:0 auto;border-radius:2px;transform:translateY(0) rotate(45deg) scale(0);animation:looping-rhombuses-spinner-animation-data-v-e013f2d4 2.5s linear infinite}.looping-rhombuses-spinner .rhombus[data-v-e013f2d4]:nth-child(1){animation-delay:calc(2500ms * 1 / -1.5)}.looping-rhombuses-spinner .rhombus[data-v-e013f2d4]:nth-child(2){animation-delay:calc(2500ms * 2 / -1.5)}.looping-rhombuses-spinner .rhombus[data-v-e013f2d4]:nth-child(3){animation-delay:calc(2500ms * 3 / -1.5)}@keyframes looping-rhombuses-spinner-animation-data-v-e013f2d4{0%{transform:translateX(0) rotate(45deg) scale(0)}50%{transform:translateX(-233%) rotate(45deg) scale(1)}100%{transform:translateX(-466%) rotate(45deg) scale(0)}}", map: undefined, media: undefined });

    };
    /* scoped */
    const __vue_scope_id__$h = "data-v-e013f2d4";
    /* module identifier */
    const __vue_module_identifier__$h = undefined;
    /* functional template */
    const __vue_is_functional_template__$h = false;
    /* style inject SSR */
    

    
    normalizeComponent_1(
      { render: __vue_render__$h, staticRenderFns: __vue_staticRenderFns__$h },
      __vue_inject_styles__$h,
      __vue_script__$h,
      __vue_scope_id__$h,
      __vue_is_functional_template__$h,
      __vue_module_identifier__$h,
      browser,
      undefined
    );

  //
  //
  //
  //
  //
  //
  //

  var script$i = {
    name: 'HalfCircleSpinner',

    props: {
      animationDuration: {
        type: Number,
        default: 1000
      },
      size: {
        type: Number,
        default: 60
      },
      color: {
        type: String,
        default: '#fff'
      }
    },

    computed: {
      spinnerStyle () {
        return {
          height: `${this.size}px`,
          width: `${this.size}px`
        }
      },

      circleStyle () {
        return {
          borderWidth: `${this.size / 10}px`,
          animationDuration: `${this.animationDuration}ms`
        }
      },

      circle1Style () {
        return Object.assign({
          borderTopColor: this.color
        }, this.circleStyle)
      },

      circle2Style () {
        return Object.assign({
          borderBottomColor: this.color
        }, this.circleStyle)
      }
    }
  };

  /* script */
  const __vue_script__$i = script$i;

  /* template */
  var __vue_render__$i = function () {var _vm=this;var _h=_vm.$createElement;var _c=_vm._self._c||_h;return _c('div',{staticClass:"half-circle-spinner",style:(_vm.spinnerStyle)},[_c('div',{staticClass:"circle circle-1",style:(_vm.circle1Style)}),_vm._v(" "),_c('div',{staticClass:"circle circle-2",style:(_vm.circle2Style)})])};
  var __vue_staticRenderFns__$i = [];

    /* style */
    const __vue_inject_styles__$i = function (inject) {
      if (!inject) return
      inject("data-v-5d773fc0_0", { source: ".half-circle-spinner[data-v-5d773fc0],.half-circle-spinner *[data-v-5d773fc0]{box-sizing:border-box}.half-circle-spinner[data-v-5d773fc0]{width:60px;height:60px;border-radius:100%;position:relative}.half-circle-spinner .circle[data-v-5d773fc0]{content:\"\";position:absolute;width:100%;height:100%;border-radius:100%;border:calc(60px / 10) solid transparent}.half-circle-spinner .circle.circle-1[data-v-5d773fc0]{border-top-color:#ff1d5e;animation:half-circle-spinner-animation-data-v-5d773fc0 1s infinite}.half-circle-spinner .circle.circle-2[data-v-5d773fc0]{border-bottom-color:#ff1d5e;animation:half-circle-spinner-animation-data-v-5d773fc0 1s infinite alternate}@keyframes half-circle-spinner-animation-data-v-5d773fc0{0%{transform:rotate(0)}100%{transform:rotate(360deg)}}", map: undefined, media: undefined });

    };
    /* scoped */
    const __vue_scope_id__$i = "data-v-5d773fc0";
    /* module identifier */
    const __vue_module_identifier__$i = undefined;
    /* functional template */
    const __vue_is_functional_template__$i = false;
    /* style inject SSR */
    

    
    normalizeComponent_1(
      { render: __vue_render__$i, staticRenderFns: __vue_staticRenderFns__$i },
      __vue_inject_styles__$i,
      __vue_script__$i,
      __vue_scope_id__$i,
      __vue_is_functional_template__$i,
      __vue_module_identifier__$i,
      browser,
      undefined
    );

  //
  //
  //
  //
  //
  //
  //
  //
  //
  //
  //
  //
  //
  //
  //

  var script$j = {
    name: 'AtomSpinner',

    props: {
      animationDuration: {
        type: Number,
        default: 1000
      },
      size: {
        type: Number,
        default: 60
      },
      color: {
        type: String,
        default: '#fff'
      }
    },

    computed: {
      spinnerStyle () {
        return {
          height: `${this.size}px`,
          width: `${this.size}px`
        }
      },

      circleStyle () {
        return {
          color: this.color,
          fontSize: `${this.size * 0.24}px`
        }
      },

      lineStyle () {
        return {
          animationDuration: `${this.animationDuration}ms`,
          borderLeftWidth: `${this.size / 25}px`,
          borderTopWidth: `${this.size / 25}px`,
          borderLeftColor: this.color
        }
      }
    }
  };

  /* script */
  const __vue_script__$j = script$j;

  /* template */
  var __vue_render__$j = function () {var _vm=this;var _h=_vm.$createElement;var _c=_vm._self._c||_h;return _c('div',{staticClass:"atom-spinner",style:(_vm.spinnerStyle)},[_c('div',{staticClass:"spinner-inner"},[_c('div',{staticClass:"spinner-line",style:(_vm.lineStyle)}),_vm._v(" "),_c('div',{staticClass:"spinner-line",style:(_vm.lineStyle)}),_vm._v(" "),_c('div',{staticClass:"spinner-line",style:(_vm.lineStyle)}),_vm._v(" "),_c('div',{staticClass:"spinner-circle",style:(_vm.circleStyle)},[_vm._v("\n      ●\n    ")])])])};
  var __vue_staticRenderFns__$j = [];

    /* style */
    const __vue_inject_styles__$j = function (inject) {
      if (!inject) return
      inject("data-v-4bc06b7a_0", { source: ".atom-spinner[data-v-4bc06b7a],.atom-spinner *[data-v-4bc06b7a]{box-sizing:border-box}.atom-spinner[data-v-4bc06b7a]{height:60px;width:60px;overflow:hidden}.atom-spinner .spinner-inner[data-v-4bc06b7a]{position:relative;display:block;height:100%;width:100%}.atom-spinner .spinner-circle[data-v-4bc06b7a]{display:block;position:absolute;color:#ff1d5e;font-size:calc(60px * .24);top:50%;left:50%;transform:translate(-50%,-50%)}.atom-spinner .spinner-line[data-v-4bc06b7a]{position:absolute;width:100%;height:100%;border-radius:50%;animation-duration:1s;border-left-width:calc(60px / 25);border-top-width:calc(60px / 25);border-left-color:#ff1d5e;border-left-style:solid;border-top-style:solid;border-top-color:transparent}.atom-spinner .spinner-line[data-v-4bc06b7a]:nth-child(1){animation:atom-spinner-animation-1-data-v-4bc06b7a 1s linear infinite;transform:rotateZ(120deg) rotateX(66deg) rotateZ(0)}.atom-spinner .spinner-line[data-v-4bc06b7a]:nth-child(2){animation:atom-spinner-animation-2-data-v-4bc06b7a 1s linear infinite;transform:rotateZ(240deg) rotateX(66deg) rotateZ(0)}.atom-spinner .spinner-line[data-v-4bc06b7a]:nth-child(3){animation:atom-spinner-animation-3-data-v-4bc06b7a 1s linear infinite;transform:rotateZ(360deg) rotateX(66deg) rotateZ(0)}@keyframes atom-spinner-animation-1-data-v-4bc06b7a{100%{transform:rotateZ(120deg) rotateX(66deg) rotateZ(360deg)}}@keyframes atom-spinner-animation-2-data-v-4bc06b7a{100%{transform:rotateZ(240deg) rotateX(66deg) rotateZ(360deg)}}@keyframes atom-spinner-animation-3-data-v-4bc06b7a{100%{transform:rotateZ(360deg) rotateX(66deg) rotateZ(360deg)}}", map: undefined, media: undefined });

    };
    /* scoped */
    const __vue_scope_id__$j = "data-v-4bc06b7a";
    /* module identifier */
    const __vue_module_identifier__$j = undefined;
    /* functional template */
    const __vue_is_functional_template__$j = false;
    /* style inject SSR */
    

    
    normalizeComponent_1(
      { render: __vue_render__$j, staticRenderFns: __vue_staticRenderFns__$j },
      __vue_inject_styles__$j,
      __vue_script__$j,
      __vue_scope_id__$j,
      __vue_is_functional_template__$j,
      __vue_module_identifier__$j,
      browser,
      undefined
    );

  //
  var script$k = {
    name: 'PopupSpinner',
    components: {
      FulfillingBouncingCircleSpinner
    },
    props: {
      loading: {
        type: Boolean,
        default: true
      },
      title: {
        type: String,
        default: ''
      },
      hasCancelButton: {
        type: Boolean,
        default: false
      },
      color: {
        type: String,
        default: '#0000ff'
      },
      size: {
        type: String,
        default: '50px'
      },
      margin: {
        type: String,
        default: '2px'
      },
      padding: {
        type: String,
        default: '15px'
      },
      radius: {
        type: String,
        default: '100%'
      }
    },

    data() {
      return {
        titleStyle: {
          textAlign: 'center'
        },
        cancelButtonStyle: {
          padding: '2px'
        },
        opened: false
      };
    },

    beforeMount() {
      // Create listener for start event.
      EventBus.$on('spinner:start', () => {
        this.show();
      }); // Create listener for stop event.

      EventBus.$on('spinner:stop', () => {
        this.hide();
      });
    },

    computed: {
      spinnerSize() {
        return parseFloat(this.size) - 25;
      },

      modalHeight() {
        // Start with the height of the spinner wrapper.
        let fullHeight = parseFloat(this.size) + 2 * parseFloat(this.padding); // If there is a title there, add space for the text.

        if (this.title !== '') {
          fullHeight = fullHeight + 20 + parseFloat(this.padding);
        } // If there is a cancel button there, add space for it.


        if (this.hasCancelButton) {
          fullHeight = fullHeight + 20 + parseFloat(this.padding);
        }

        return fullHeight + 'px';
      },

      modalWidth() {
        return parseFloat(this.size) + 2 * parseFloat(this.padding) + 'px';
      }

    },
    methods: {
      beforeOpen() {
        window.addEventListener('keyup', this.onKey);
        this.opened = true;
      },

      beforeClose() {
        window.removeEventListener('keyup', this.onKey);
        this.opened = false;
      },

      onKey(event) {
        if (event.keyCode == 27) {
          console.log('Exited spinner through Esc key');
          this.cancel();
        }
      },

      cancel() {
        this.$emit('spinner-cancel');
        this.hide();
      },

      show() {
        this.$modal.show('popup-spinner'); // Bring up the spinner modal.
      },

      hide() {
        this.$modal.hide('popup-spinner'); // Dispel the spinner modal.
      }

    }
  };

  /* script */
  const __vue_script__$k = script$k;

  /* template */
  var __vue_render__$k = function () {var _vm=this;var _h=_vm.$createElement;var _c=_vm._self._c||_h;return _c('modal',{staticStyle:{"opacity":"1.0"},attrs:{"name":"popup-spinner","height":_vm.modalHeight,"width":_vm.modalWidth,"click-to-close":false},on:{"before-open":_vm.beforeOpen,"before-close":_vm.beforeClose}},[_c('div',{staticClass:"overlay-box"},[_c('div',{staticClass:"loader-box"},[_c('fulfilling-bouncing-circle-spinner',{attrs:{"color":_vm.color,"size":_vm.spinnerSize,"animation-duration":2000}})],1),_vm._v(" "),(_vm.title !== '')?_c('div',{style:(_vm.titleStyle)},[_vm._v("\n      "+_vm._s(_vm.title)+"\n    ")]):_vm._e(),_vm._v(" "),(_vm.hasCancelButton)?_c('div',{staticStyle:{"padding":"13px"}},[_c('button',{style:(_vm.cancelButtonStyle),on:{"click":_vm.cancel}},[_vm._v("Cancel")])]):_vm._e()])])};
  var __vue_staticRenderFns__$k = [];

    /* style */
    const __vue_inject_styles__$k = function (inject) {
      if (!inject) return
      inject("data-v-3bb427d4_0", { source: ".loader-box[data-v-3bb427d4]{display:flex;justify-content:center}.overlay-box[data-v-3bb427d4]{display:flex;flex-direction:column;height:100%;justify-content:space-evenly}", map: undefined, media: undefined });

    };
    /* scoped */
    const __vue_scope_id__$k = "data-v-3bb427d4";
    /* module identifier */
    const __vue_module_identifier__$k = undefined;
    /* functional template */
    const __vue_is_functional_template__$k = false;
    /* style inject SSR */
    

    
    var PopupSpinner = normalizeComponent_1(
      { render: __vue_render__$k, staticRenderFns: __vue_staticRenderFns__$k },
      __vue_inject_styles__$k,
      __vue_script__$k,
      __vue_scope_id__$k,
      __vue_is_functional_template__$k,
      __vue_module_identifier__$k,
      browser,
      undefined
    );

  //
  //
  //
  //
  //
  //
  //
  //
  //
  //
  //
  //
  //
  //
  //
  //
  //
  //
  //
  //
  //
  //
  //
  //
  //
  //
  //
  //
  //
  //
  //
  //
  //
  //
  //
  //
  //
  //
  //
  //
  var script$l = {
    name: 'notification',
    props: {
      message: String,
      icon: {
        type: String,
        default: 'ti-info-alt'
      },
      verticalAlign: {
        type: String,
        default: 'top'
      },
      horizontalAlign: {
        type: String,
        default: 'right'
      },
      type: {
        type: String,
        default: 'info'
      },
      timeout: {
        type: Number,
        default: 2000
      },
      timestamp: {
        type: Date,
        default: () => new Date()
      }
    },

    data() {
      return {};
    },

    computed: {
      hasIcon() {
        return this.icon && this.icon.length > 0;
      },

      alertType() {
        return `alert-${this.type}`;
      },

      customPosition() {
        let initialMargin = 20;
        let alertHeight = 60;
        let sameAlertsCount = this.$notifications.state.filter(alert => {
          return alert.horizontalAlign === this.horizontalAlign && alert.verticalAlign === this.verticalAlign;
        }).length;
        let pixels = (sameAlertsCount - 1) * alertHeight + initialMargin;
        let styles = {};

        if (this.verticalAlign === 'top') {
          styles.top = `${pixels}px`;
        } else {
          styles.bottom = `${pixels}px`;
        }

        return styles;
      }

    },
    methods: {
      close() {
        this.$parent.$emit('on-close', this.timestamp);
      }

    },

    mounted() {
      if (this.timeout) {
        setTimeout(this.close, this.timeout);
      }
    }

  };

  /* script */
  const __vue_script__$l = script$l;

  /* template */
  var __vue_render__$l = function () {var _vm=this;var _h=_vm.$createElement;var _c=_vm._self._c||_h;return _c('div',{staticClass:"alert open alert-with-icon",class:[_vm.verticalAlign, _vm.horizontalAlign, _vm.alertType],style:(_vm.customPosition),attrs:{"role":"alert","data-notify":"container","data-notify-position":"top-center"}},[_c('div',{staticClass:"notification-box"},[_c('div',[_c('span',{staticClass:"alert-icon",class:_vm.icon,attrs:{"data-notify":"message"}})]),_vm._v(" "),_c('div',{staticClass:"message-box"},[_c('div',{staticClass:"message",attrs:{"data-notify":"message"},domProps:{"innerHTML":_vm._s(_vm.message)}})]),_vm._v(" "),_c('div',[_c('button',{staticClass:"btn__trans close-button",attrs:{"aria-hidden":"true","data-notify":"dismiss"},on:{"click":_vm.close}},[_c('i',{staticClass:"ti-close"})])])])])};
  var __vue_staticRenderFns__$l = [];

    /* style */
    const __vue_inject_styles__$l = function (inject) {
      if (!inject) return
      inject("data-v-61be677e_0", { source: "@import url(https://cdn.jsdelivr.net/gh/lykmapipo/themify-icons@0.1.2/css/themify-icons.css);", map: undefined, media: undefined })
  ,inject("data-v-61be677e_1", { source: ".fade-enter-active[data-v-61be677e],.fade-leave-active[data-v-61be677e]{transition:opacity .3s}.fade-enter[data-v-61be677e],.fade-leave-to[data-v-61be677e]{opacity:0}.close-button[data-v-61be677e],.close-button[data-v-61be677e]:hover{background:0 0;line-height:0;padding:5px 5px;margin-left:10px;border-radius:3px}.close-button[data-v-61be677e]:hover{background:#ffffff63;color:#737373}.alert[data-v-61be677e]{border:0;border-radius:0;color:#fff;padding:20px 15px;font-size:14px;z-index:100;display:inline-block;position:fixed;transition:all .5s ease-in-out}.container .alert[data-v-61be677e]{border-radius:4px}.alert.center[data-v-61be677e]{left:0;right:0;margin:0 auto}.alert.left[data-v-61be677e]{left:20px}.alert.right[data-v-61be677e]{right:20px}.container .alert[data-v-61be677e]{border-radius:0}.alert .alert-icon[data-v-61be677e]{font-size:30px;margin-right:5px}.alert .close~span[data-v-61be677e]{display:inline-block;max-width:89%}.alert[data-notify=container][data-v-61be677e]{padding:0;border-radius:2px}.alert span[data-notify=icon][data-v-61be677e]{font-size:30px;display:block;left:15px;position:absolute;top:50%;margin-top:-20px}.alert-info[data-v-61be677e]{background-color:#7ce4fe;color:#3091b2}.alert-success[data-v-61be677e]{background-color:#080;color:#fff}.alert-warning[data-v-61be677e]{background-color:#e29722;color:#fff}.alert-danger[data-v-61be677e]{background-color:#ff8f5e;color:#b33c12}.message-box[data-v-61be677e]{font-size:15px;align-content:center;max-width:400px;min-width:150px;padding-left:10px;flex-grow:1}.message-box .message[data-v-61be677e]{line-height:1.5em;width:100%}.notification-box[data-v-61be677e]{display:flex;justify-content:flex-start;padding:10px 15px}.notification-box>div[data-v-61be677e]{align-self:center}.btn__trans[data-v-61be677e]{font-size:18px;color:#fff;background-color:transparent;background-repeat:no-repeat;border:none;cursor:pointer;overflow:hidden;background-image:none;outline:0}", map: undefined, media: undefined });

    };
    /* scoped */
    const __vue_scope_id__$l = "data-v-61be677e";
    /* module identifier */
    const __vue_module_identifier__$l = undefined;
    /* functional template */
    const __vue_is_functional_template__$l = false;
    /* style inject SSR */
    

    
    var Notification = normalizeComponent_1(
      { render: __vue_render__$l, staticRenderFns: __vue_staticRenderFns__$l },
      __vue_inject_styles__$l,
      __vue_script__$l,
      __vue_scope_id__$l,
      __vue_is_functional_template__$l,
      __vue_module_identifier__$l,
      browser,
      undefined
    );

  //
  var script$m = {
    components: {
      Notification
    },

    data() {
      return {
        notifications: this.$notifications.state
      };
    },

    methods: {
      removeNotification(timestamp) {
        this.$notifications.removeNotification(timestamp);
      },

      clearAllNotifications() {
        this.$notifications.clear();
      }

    }
  };

  /* script */
  const __vue_script__$m = script$m;

  /* template */
  var __vue_render__$m = function () {var _vm=this;var _h=_vm.$createElement;var _c=_vm._self._c||_h;return _c('div',{staticClass:"notifications"},[_c('transition-group',{attrs:{"name":"list"},on:{"on-close":_vm.removeNotification}},_vm._l((_vm.notifications),function(notification,index){return _c('notification',{key:index+0,attrs:{"message":notification.message,"icon":notification.icon,"type":notification.type,"vertical-align":notification.verticalAlign,"horizontal-align":notification.horizontalAlign,"timeout":notification.timeout,"timestamp":notification.timestamp}})}),1)],1)};
  var __vue_staticRenderFns__$m = [];

    /* style */
    const __vue_inject_styles__$m = function (inject) {
      if (!inject) return
      inject("data-v-2a7eddb0_0", { source: ".list-item{display:inline-block;margin-right:10px}.list-enter-active,.list-leave-active{transition:all 1s}.list-enter,.list-leave-to{opacity:0;transform:translateY(-30px)}", map: undefined, media: undefined });

    };
    /* scoped */
    const __vue_scope_id__$m = undefined;
    /* module identifier */
    const __vue_module_identifier__$m = undefined;
    /* functional template */
    const __vue_is_functional_template__$m = false;
    /* style inject SSR */
    

    
    var Notifications = normalizeComponent_1(
      { render: __vue_render__$m, staticRenderFns: __vue_staticRenderFns__$m },
      __vue_inject_styles__$m,
      __vue_script__$m,
      __vue_scope_id__$m,
      __vue_is_functional_template__$m,
      __vue_module_identifier__$m,
      browser,
      undefined
    );

  //
  //
  //
  //
  //
  //
  //
  //
  //
  //
  //
  //
  //
  //
  //
  //
  //
  //
  //
  //
  //
  //
  //
  //
  //
  //
  var script$n = {
    props: {
      title: String,
      icon: String,
      width: {
        type: String,
        default: "170px"
      }
    },

    data() {
      return {
        isOpen: false
      };
    },

    computed: {
      style() {
        return 'width: ' + this.width;
      }

    },
    methods: {
      toggleDropDown() {
        this.isOpen = !this.isOpen;
      },

      closeDropDown() {
        this.isOpen = false;
      }

    }
  };

  /* script */
  const __vue_script__$n = script$n;

  /* template */
  var __vue_render__$n = function () {var _vm=this;var _h=_vm.$createElement;var _c=_vm._self._c||_h;return _c('li',{directives:[{name:"click-outside",rawName:"v-click-outside",value:(_vm.closeDropDown),expression:"closeDropDown"}],staticClass:"dropdown",class:{open:_vm.isOpen}},[_c('a',{staticClass:"dropdown-toggle btn-rotate",style:(_vm.style),attrs:{"href":"javascript:void(0)","data-toggle":"dropdown"},on:{"click":_vm.toggleDropDown}},[_vm._t("title",[_c('i',{class:_vm.icon}),_vm._v(" "),_c('div',{staticClass:"dropdown-title"},[_vm._v(_vm._s(_vm.title)+"\n        "),_c('b',{staticClass:"caret"})])])],2),_vm._v(" "),_c('ul',{staticClass:"dropdown-menu"},[_vm._t("default")],2)])};
  var __vue_staticRenderFns__$n = [];

    /* style */
    const __vue_inject_styles__$n = function (inject) {
      if (!inject) return
      inject("data-v-73a696f8_0", { source: ".dropdown-toggle{cursor:pointer;display:flex;justify-content:space-evenly;text-transform:initial}.dropdown-toggle:after{position:absolute;right:10px;top:50%;margin-top:-2px}.dropdown-menu{margin-top:20px}", map: undefined, media: undefined });

    };
    /* scoped */
    const __vue_scope_id__$n = undefined;
    /* module identifier */
    const __vue_module_identifier__$n = undefined;
    /* functional template */
    const __vue_is_functional_template__$n = false;
    /* style inject SSR */
    

    
    var Dropdown = normalizeComponent_1(
      { render: __vue_render__$n, staticRenderFns: __vue_staticRenderFns__$n },
      __vue_inject_styles__$n,
      __vue_script__$n,
      __vue_scope_id__$n,
      __vue_is_functional_template__$n,
      __vue_module_identifier__$n,
      browser,
      undefined
    );

  const NotificationStore = {
    state: [],

    // here the notifications will be added
    removeNotification(timestamp) {
      const indexToDelete = this.state.findIndex(n => n.timestamp === timestamp);

      if (indexToDelete !== -1) {
        this.state.splice(indexToDelete, 1);
      }
    },

    notify(notification) {
      // Create a timestamp to serve as a unique ID for the notification.
      notification.timestamp = new Date();
      notification.timestamp.setMilliseconds(notification.timestamp.getMilliseconds() + this.state.length);
      this.state.push(notification);
    },

    clear() {
      // This removes all of them in a way that the GUI keeps up.
      while (this.state.length > 0) {
        this.removeNotification(this.state[0].timestamp);
      }
    }

  };

  function setupSpinner(Vue) {
    // Create the global $spinner functions the user can call 
    // from inside any component.
    Vue.prototype.$spinner = {
      start() {
        // Send a start event to the bus.
        EventBus.$emit('spinner:start');
      },

      stop() {
        // Send a stop event to the bus.
        EventBus.$emit('spinner:stop');
      }

    };
  }

  function setupNotifications(Vue) {
    Object.defineProperty(Vue.prototype, '$notifications', {
      get() {
        return NotificationStore;
      }

    });
  }

  function setupProgressBar(Vue, options) {
    Vue.use(vueProgressbar, options);
  }

  function install(Vue, options = {}) {
    Vue.use(VModal);

    if (!options.notifications || !options.notifications.disabled) {
      setupNotifications(Vue);
      Vue.component('Notifications', Notifications);
    }

    if ((!options.spinner || !options.spinner.disabled) && !this.spinnerInstalled) {
      this.spinnerInstalled = true;
      setupSpinner(Vue);
      Vue.component('PopupSpinner', PopupSpinner);
    }

    if (!options.progressbar || !options.progressbar.disabled) {
      var progressbarOptions = options.progressbar ? options.progressbar.options : {};
      setupProgressBar(Vue, progressbarOptions);
    }

    Vue.component('Dropdown', Dropdown);
    Vue.component('DialogDrag', DialogDrag);
    Vue.directive('click-outside', directive_1);
  } // Automatic installation if Vue has been added to the global scope.


  if (typeof window !== 'undefined' && window.Vue) {
    window.Vue.use({
      install
    });
  }

  var ScirisVue = {
    install
  };

  const rpc = rpcs.rpc;
  const download = rpcs.download;
  const upload = rpcs.upload;
  const succeed$1 = status.succeed;
  const fail$1 = status.fail;
  const start$1 = status.start;
  const notify$1 = status.notify;
  const placeholders$1 = graphs.placeholders;
  const clearGraphs$1 = graphs.clearGraphs;
  const makeGraphs$1 = graphs.makeGraphs;
  const scaleFigs$1 = graphs.scaleFigs;
  const showBrowserWindowSize$1 = graphs.showBrowserWindowSize;
  const addListener$1 = graphs.addListener;
  const onMouseUpdate$1 = graphs.onMouseUpdate;
  const createDialogs$1 = graphs.createDialogs;
  const newDialog$1 = graphs.newDialog;
  const findDialog$1 = graphs.findDialog;
  const maximize$1 = graphs.maximize;
  const minimize$1 = graphs.minimize;
  const mpld3$1 = graphs.mpld3;
  let draw_figure = null;

  if (mpld3$1 !== null) {
    draw_figure = mpld3$1.draw_figure;
  }

  const getTaskResultWaiting$1 = tasks.getTaskResultWaiting;
  const getTaskResultPolling$1 = tasks.getTaskResultPolling;
  const loginCall$1 = user.loginCall;
  const logoutCall$1 = user.logoutCall;
  const getCurrentUserInfo$1 = user.getCurrentUserInfo;
  const registerUser$1 = user.registerUser;
  const changeUserInfo$1 = user.changeUserInfo;
  const changeUserPassword$1 = user.changeUserPassword;
  const adminGetUserInfo$1 = user.adminGetUserInfo;
  const deleteUser$1 = user.deleteUser;
  const activateUserAccount$1 = user.activateUserAccount;
  const deactivateUserAccount$1 = user.deactivateUserAccount;
  const grantUserAdminRights$1 = user.grantUserAdminRights;
  const revokeUserAdminRights$1 = user.revokeUserAdminRights;
  const resetUserPassword$1 = user.resetUserPassword;
  const getUserInfo$1 = user.getUserInfo;
  const currentUser = user.currentUser;
  const checkLoggedIn$1 = user.checkLoggedIn;
  const checkAdminLoggedIn$1 = user.checkAdminLoggedIn;
  const logOut = user.logOut;
  const sleep$1 = utils.sleep;
  const getUniqueName$1 = utils.getUniqueName;
  const sciris = {
    // rpc-service.js
    rpc,
    download,
    upload,
    // graphs.js
    placeholders: placeholders$1,
    clearGraphs: clearGraphs$1,
    makeGraphs: makeGraphs$1,
    scaleFigs: scaleFigs$1,
    showBrowserWindowSize: showBrowserWindowSize$1,
    addListener: addListener$1,
    onMouseUpdate: onMouseUpdate$1,
    createDialogs: createDialogs$1,
    newDialog: newDialog$1,
    findDialog: findDialog$1,
    maximize: maximize$1,
    minimize: minimize$1,
    mpld3: mpld3$1,
    draw_figure,
    // status-service.js
    succeed: succeed$1,
    fail: fail$1,
    start: start$1,
    notify: notify$1,
    // task-service.js
    getTaskResultWaiting: getTaskResultWaiting$1,
    getTaskResultPolling: getTaskResultPolling$1,
    // user-service.js
    loginCall: loginCall$1,
    logoutCall: logoutCall$1,
    getCurrentUserInfo: getCurrentUserInfo$1,
    registerUser: registerUser$1,
    changeUserInfo: changeUserInfo$1,
    changeUserPassword: changeUserPassword$1,
    adminGetUserInfo: adminGetUserInfo$1,
    deleteUser: deleteUser$1,
    activateUserAccount: activateUserAccount$1,
    deactivateUserAccount: deactivateUserAccount$1,
    grantUserAdminRights: grantUserAdminRights$1,
    revokeUserAdminRights: revokeUserAdminRights$1,
    resetUserPassword: resetUserPassword$1,
    getUserInfo: getUserInfo$1,
    currentUser,
    checkLoggedIn: checkLoggedIn$1,
    checkAdminLoggedIn: checkAdminLoggedIn$1,
    logOut,
    // utils.js
    sleep: sleep$1,
    getUniqueName: getUniqueName$1,
    rpcs,
    graphs,
    status,
    user,
    tasks,
    utils,
    ScirisVue,
    EventBus
  };

  exports.default = sciris;
  exports.sciris = sciris;

  Object.defineProperty(exports, '__esModule', { value: true });

})));

}).call(this,require('_process'),typeof global !== "undefined" ? global : typeof self !== "undefined" ? self : typeof window !== "undefined" ? window : {},require("timers").setImmediate)
},{"_process":4,"mpld3":3,"timers":5}],2:[function(require,module,exports){
module.exports = require('./dist/sciris-js.js').default;

},{"./dist/sciris-js.js":1}],3:[function(require,module,exports){
(function (global){
(function(f){if(typeof exports==="object"&&typeof module!=="undefined"){module.exports=f()}else if(typeof define==="function"&&define.amd){define([],f)}else{var g;if(typeof window!=="undefined"){g=window}else if(typeof global!=="undefined"){g=global}else if(typeof self!=="undefined"){g=self}else{g=this}g.mpld3 = f()}})(function(){var define,module,exports;return (function(){function r(e,n,t){function o(i,f){if(!n[i]){if(!e[i]){var c="function"==typeof require&&require;if(!f&&c)return c(i,!0);if(u)return u(i,!0);var a=new Error("Cannot find module '"+i+"'");throw a.code="MODULE_NOT_FOUND",a}var p=n[i]={exports:{}};e[i][0].call(p.exports,function(r){var n=e[i][1][r];return o(n||r)},p,p.exports,r,e,n,t)}return n[i].exports}for(var u="function"==typeof require&&require,i=0;i<t.length;i++)o(t[i]);return o}return r})()({1:[function(require,module,exports){
!function(t){function s(t){var s={};for(var i in t)s[i]=t[i];return s}function i(t,s){t="undefined"!=typeof t?t:10,s="undefined"!=typeof s?s:"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";for(var i=s.charAt(Math.round(Math.random()*(s.length-11))),e=1;t>e;e++)i+=s.charAt(Math.round(Math.random()*(s.length-1)));return i}function e(s,i){var e=t.interpolate([s[0].valueOf(),s[1].valueOf()],[i[0].valueOf(),i[1].valueOf()]);return function(t){var s=e(t);return[new Date(s[0]),new Date(s[1])]}}function o(t){return"undefined"==typeof t}function r(t){return null==t||o(t)}function n(t,s){return t.length>0?t[s%t.length]:null}function a(t){function s(t,s){var n=function(t){return"function"==typeof t?t:function(){return t}},a=n(i),p=n(e),h=[],l=[],c=0,u=-1,d=0,f=!1;if(!s){s=["M"];for(var y=1;y<t.length;y++)s.push("L")}for(;++u<s.length;){for(d=c+r[s[u]],h=[];d>c;)o.call(this,t[c],c)?(h.push(a.call(this,t[c],c),p.call(this,t[c],c)),c++):(h=null,c=d);h?f&&h.length>0?(l.push("M",h[0],h[1]),f=!1):(l.push(s[u]),l=l.concat(h)):f=!0}return c!=t.length&&console.warn("Warning: not all vertices used in Path"),l.join(" ")}var i=function(t,s){return t[0]},e=function(t,s){return t[1]},o=function(t,s){return!0},r={M:1,m:1,L:1,l:1,Q:2,q:2,T:1,t:1,S:2,s:2,C:3,c:3,Z:0,z:0};return s.x=function(t){return arguments.length?(i=t,s):i},s.y=function(t){return arguments.length?(e=t,s):e},s.defined=function(t){return arguments.length?(o=t,s):o},s.call=s,s}function p(t){function s(t){return i.forEach(function(s){t=s(t)}),t}var i=Array.prototype.slice.call(arguments,0),e=i.length;return s.domain=function(t){return arguments.length?(i[0].domain(t),s):i[0].domain()},s.range=function(t){return arguments.length?(i[e-1].range(t),s):i[e-1].range()},s.step=function(t){return i[t]},s}function h(t,s){if(F.call(this,t,s),this.cssclass="mpld3-"+this.props.xy+"grid","x"==this.props.xy)this.transform="translate(0,"+this.ax.height+")",this.position="bottom",this.scale=this.ax.xdom,this.tickSize=-this.ax.height;else{if("y"!=this.props.xy)throw"unrecognized grid xy specifier: should be 'x' or 'y'";this.transform="translate(0,0)",this.position="left",this.scale=this.ax.ydom,this.tickSize=-this.ax.width}}function l(t,s){F.call(this,t,s);var i={bottom:[0,this.ax.height],top:[0,0],left:[0,0],right:[this.ax.width,0]},e={bottom:"x",top:"x",left:"y",right:"y"};this.ax=t,this.transform="translate("+i[this.props.position]+")",this.props.xy=e[this.props.position],this.cssclass="mpld3-"+this.props.xy+"axis",this.scale=this.ax[this.props.xy+"dom"],this.tickNr=null,this.tickFormat=null}function c(t,s){if(this.trans=t,"undefined"==typeof s){if(this.ax=null,this.fig=null,"display"!==this.trans)throw"ax must be defined if transform != 'display'"}else this.ax=s,this.fig=s.fig;if(this.zoomable="data"===this.trans,this.x=this["x_"+this.trans],this.y=this["y_"+this.trans],"undefined"==typeof this.x||"undefined"==typeof this.y)throw"unrecognized coordinate code: "+this.trans}function u(t,s){F.call(this,t,s),this.data=t.fig.get_data(this.props.data),this.pathcodes=this.props.pathcodes,this.pathcoords=new c(this.props.coordinates,this.ax),this.offsetcoords=new c(this.props.offsetcoordinates,this.ax),this.datafunc=a()}function d(t,s){F.call(this,t,s),(null==this.props.facecolors||0==this.props.facecolors.length)&&(this.props.facecolors=["none"]),(null==this.props.edgecolors||0==this.props.edgecolors.length)&&(this.props.edgecolors=["none"]);var i=this.ax.fig.get_data(this.props.offsets);(null===i||0===i.length)&&(i=[null]);var e=Math.max(this.props.paths.length,i.length);if(i.length===e)this.offsets=i;else{this.offsets=[];for(var o=0;e>o;o++)this.offsets.push(n(i,o))}this.pathcoords=new c(this.props.pathcoordinates,this.ax),this.offsetcoords=new c(this.props.offsetcoordinates,this.ax)}function f(s,i){F.call(this,s,i);var e=this.props;e.facecolor="none",e.edgecolor=e.color,delete e.color,e.edgewidth=e.linewidth,delete e.linewidth;const o=e.drawstyle;switch(delete e.drawstyle,this.defaultProps=u.prototype.defaultProps,u.call(this,s,e),o){case"steps":case"steps-pre":this.datafunc=t.line().curve(t.curveStepBefore);break;case"steps-post":this.datafunc=t.line().curve(t.curveStepAfter);break;case"steps-mid":this.datafunc=t.line().curve(t.curveStep);break;default:this.datafunc=t.line().curve(t.curveLinear)}}function y(s,i){F.call(this,s,i),null!==this.props.markerpath?this.marker=0==this.props.markerpath[0].length?null:T.path().call(this.props.markerpath[0],this.props.markerpath[1]):this.marker=null===this.props.markername?null:t.symbol(this.props.markername).size(Math.pow(this.props.markersize,2))();var e={paths:[this.props.markerpath],offsets:s.fig.parse_offsets(s.fig.get_data(this.props.data,!0)),xindex:this.props.xindex,yindex:this.props.yindex,offsetcoordinates:this.props.coordinates,edgecolors:[this.props.edgecolor],edgewidths:[this.props.edgewidth],facecolors:[this.props.facecolor],alphas:[this.props.alpha],zorder:this.props.zorder,id:this.props.id};this.requiredProps=d.prototype.requiredProps,this.defaultProps=d.prototype.defaultProps,d.call(this,s,e)}function m(t,s){F.call(this,t,s),this.coords=new c(this.props.coordinates,this.ax)}function g(t,s){F.call(this,t,s),this.text=this.props.text,this.position=this.props.position,this.coords=new c(this.props.coordinates,this.ax)}function x(s,i){function e(t){return new Date(t[0],t[1],t[2],t[3],t[4],t[5])}function o(t,s){return"date"!==t?s:[e(s[0]),e(s[1])]}function r(s,i,e){var o="date"===s?t.scaleTime():"log"===s?t.scaleLog():t.scaleLinear();return o.domain(i).range(e)}F.call(this,s,i),this.axnum=this.fig.axes.length,this.axid=this.fig.figid+"_ax"+(this.axnum+1),this.clipid=this.axid+"_clip",this.props.xdomain=this.props.xdomain||this.props.xlim,this.props.ydomain=this.props.ydomain||this.props.ylim,this.sharex=[],this.sharey=[],this.elements=[],this.axisList=[];var n=this.props.bbox;this.position=[n[0]*this.fig.width,(1-n[1]-n[3])*this.fig.height],this.width=n[2]*this.fig.width,this.height=n[3]*this.fig.height,this.isZoomEnabled=null,this.zoom=null,this.lastTransform=t.zoomIdentity,this.isBoxzoomEnabled=null,this.isLinkedBrushEnabled=null,this.isCurrentLinkedBrushTarget=!1,this.brushG=null,this.props.xdomain=o(this.props.xscale,this.props.xdomain),this.props.ydomain=o(this.props.yscale,this.props.ydomain),this.x=this.xdom=r(this.props.xscale,this.props.xdomain,[0,this.width]),this.y=this.ydom=r(this.props.yscale,this.props.ydomain,[this.height,0]),"date"===this.props.xscale&&(this.x=T.multiscale(t.scaleLinear().domain(this.props.xlim).range(this.props.xdomain.map(Number)),this.xdom)),"date"===this.props.yscale&&(this.y=T.multiscale(t.scaleLinear().domain(this.props.ylim).range(this.props.ydomain.map(Number)),this.ydom));for(var a=this.props.axes,p=0;p<a.length;p++){var h=new T.Axis(this,a[p]);this.axisList.push(h),this.elements.push(h),(this.props.gridOn||h.props.grid.gridOn)&&this.elements.push(h.getGrid())}for(var l=this.props.paths,p=0;p<l.length;p++)this.elements.push(new T.Path(this,l[p]));for(var c=this.props.lines,p=0;p<c.length;p++)this.elements.push(new T.Line(this,c[p]));for(var u=this.props.markers,p=0;p<u.length;p++)this.elements.push(new T.Markers(this,u[p]));for(var d=this.props.texts,p=0;p<d.length;p++)this.elements.push(new T.Text(this,d[p]));for(var f=this.props.collections,p=0;p<f.length;p++)this.elements.push(new T.PathCollection(this,f[p]));for(var y=this.props.images,p=0;p<y.length;p++)this.elements.push(new T.Image(this,y[p]));this.elements.sort(function(t,s){return t.props.zorder-s.props.zorder})}function b(t,s){F.call(this,t,s),this.buttons=[],this.props.buttons.forEach(this.addButton.bind(this))}function v(t,s){F.call(this,t),this.toolbar=t,this.fig=this.toolbar.fig,this.cssclass="mpld3-"+s+"button",this.active=!1}function A(t,s){F.call(this,t,s)}function k(t,s){A.call(this,t,s);var i=T.ButtonFactory({buttonID:"reset",sticky:!1,onActivate:function(){this.toolbar.fig.reset()},icon:function(){return T.icons.reset}});this.fig.buttons.push(i)}function w(t,s){A.call(this,t,s),null===this.props.enabled&&(this.props.enabled=!this.props.button);var i=this.props.enabled;if(this.props.button){var e=T.ButtonFactory({buttonID:"zoom",sticky:!0,actions:["scroll","drag"],onActivate:this.activate.bind(this),onDeactivate:this.deactivate.bind(this),onDraw:function(){this.setState(i)},icon:function(){return T.icons.move}});this.fig.buttons.push(e)}}function B(t,s){A.call(this,t,s),null===this.props.enabled&&(this.props.enabled=!this.props.button);var i=this.props.enabled;if(this.props.button){var e=T.ButtonFactory({buttonID:"boxzoom",sticky:!0,actions:["drag"],onActivate:this.activate.bind(this),onDeactivate:this.deactivate.bind(this),onDraw:function(){this.setState(i)},icon:function(){return T.icons.zoom}});this.fig.buttons.push(e)}this.extentClass="boxzoombrush"}function z(t,s){A.call(this,t,s)}function E(t,s){T.Plugin.call(this,t,s),null===this.props.enabled&&(this.props.enabled=!this.props.button);var i=this.props.enabled;if(this.props.button){var e=T.ButtonFactory({buttonID:"linkedbrush",sticky:!0,actions:["drag"],onActivate:this.activate.bind(this),onDeactivate:this.deactivate.bind(this),onDraw:function(){this.setState(i)},icon:function(){return T.icons.brush}});this.fig.buttons.push(e)}this.pathCollectionsByAxes=[],this.objectsByAxes=[],this.allObjects=[],this.extentClass="linkedbrush",this.dataKey="offsets",this.objectClass=null}function P(t,s){T.Plugin.call(this,t,s)}function O(s,i){F.call(this,null,i),this.figid=s,this.width=this.props.width,this.height=this.props.height,this.data=this.props.data,this.buttons=[],this.root=t.select("#"+s).append("div").style("position","relative"),this.axes=[];for(var e=0;e<this.props.axes.length;e++)this.axes.push(new x(this,this.props.axes[e]));this.plugins=[],this.pluginsByType={},this.props.plugins.forEach(function(t){this.addPlugin(t)}.bind(this)),this.toolbar=new T.Toolbar(this,{buttons:this.buttons})}function F(t,s){this.parent=r(t)?null:t,this.props=r(s)?{}:this.processProps(s),this.fig=t instanceof O?t:t&&"fig"in t?t.fig:null,this.ax=t instanceof x?t:t&&"ax"in t?t.ax:null}var T={_mpld3IsLoaded:!0,figures:[],plugin_map:{}};T.version="0.4.13",T.register_plugin=function(t,s){T.plugin_map[t]=s},T.remove_figure=function(t){var s=document.getElementById(t);null!==s&&(s.innerHTML="");for(var i=0;i<T.figures.length;i++){var e=T.figures[i];e.figid===t&&T.figures.splice(i,1)}return!0},T.draw_figure=function(t,s,i,e){var o=document.getElementById(t);if(e="undefined"!=typeof e?e:!1,e&&T.remove_figure(t),null===o)throw t+" is not a valid id";var r=new T.Figure(t,s);return i&&i(r,o),T.figures.push(r),r.draw(),r},T.cloneObj=s,T.boundsToTransform=function(t,s){var i=t.width,e=t.height,o=s[1][0]-s[0][0],r=s[1][1]-s[0][1],n=(s[0][0]+s[1][0])/2,a=(s[0][1]+s[1][1])/2,p=Math.max(1,Math.min(8,.9/Math.max(o/i,r/e))),h=[i/2-p*n,e/2-p*a];return{translate:h,scale:p}},T.getTransformation=function(t){var s=document.createElementNS("http://www.w3.org/2000/svg","g");s.setAttributeNS(null,"transform",t);var i,e,o,r=s.transform.baseVal.consolidate().matrix,n=r.a,a=r.b,p=r.c,h=r.d,l=r.e,c=r.f;(i=Math.sqrt(n*n+a*a))&&(n/=i,a/=i),(o=n*p+a*h)&&(p-=n*o,h-=a*o),(e=Math.sqrt(p*p+h*h))&&(p/=e,h/=e,o/=e),a*p>n*h&&(n=-n,a=-a,o=-o,i=-i);var u={translateX:l,translateY:c,rotate:180*Math.atan2(a,n)/Math.PI,skewX:180*Math.atan(o)/Math.PI,scaleX:i,scaleY:e},d="translate("+u.translateX+","+u.translateY+")rotate("+u.rotate+")skewX("+u.skewX+")scale("+u.scaleX+","+u.scaleY+")";return d},T.merge_objects=function(t){for(var s,i={},e=0;e<arguments.length;e++){s=arguments[e];for(var o in s)i[o]=s[o]}return i},T.generate_id=function(t,s){return console.warn("mpld3.generate_id is deprecated. Use mpld3.generateId instead."),i(t,s)},T.generateId=i,T.get_element=function(t,s){var i,e,o;i="undefined"==typeof s?T.figures:"undefined"==typeof s.length?[s]:s;for(var r=0;r<i.length;r++){if(s=i[r],s.props.id===t)return s;for(var n=0;n<s.axes.length;n++){if(e=s.axes[n],e.props.id===t)return e;for(var a=0;a<e.elements.length;a++)if(o=e.elements[a],o.props.id===t)return o}}return null},T.insert_css=function(t,s){var i=document.head||document.getElementsByTagName("head")[0],e=document.createElement("style"),o=t+" {";for(var r in s)o+=r+":"+s[r]+"; ";o+="}",e.type="text/css",e.styleSheet?e.styleSheet.cssText=o:e.appendChild(document.createTextNode(o)),i.appendChild(e)},T.process_props=function(t,s,i,e){function o(t){F.call(this,null,t)}console.warn("mpld3.process_props is deprecated. Plot elements should derive from mpld3.PlotElement"),o.prototype=Object.create(F.prototype),o.prototype.constructor=o,o.prototype.requiredProps=e,o.prototype.defaultProps=i;var r=new o(s);return r.props},T.interpolateDates=e,T.path=function(){return a()},T.multiscale=p,T.icons={reset:"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAABmJLR0QA/wD/AP+gvaeTAAAACXBI\nWXMAAAsTAAALEwEAmpwYAAAAB3RJTUUH3gIcACMoD/OzIwAAAJhJREFUOMtjYKAx4KDUgNsMDAx7\nyNV8i4GB4T8U76VEM8mGYNNMtCH4NBM0hBjNMIwSsMzQ0MamcDkDA8NmQi6xggpUoikwQbIkHk2u\nE0rLI7vCBknBSyxeRDZAE6qHgQkq+ZeBgYERSfFPAoHNDNUDN4BswIRmKgxwEasP2dlsDAwMYlA/\n/mVgYHiBpkkGKscIDaPfVMmuAGnOTaGsXF0MAAAAAElFTkSuQmCC\n",move:"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAABmJLR0QA/wD/AP+gvaeTAAAACXBI\nWXMAAAsTAAALEwEAmpwYAAAAB3RJTUUH3gIcACQMfLHBNQAAANZJREFUOMud07FKA0EQBuAviaKB\nlFr7COJrpAyYRlKn8hECEkFEn8ROCCm0sBMRYgh5EgVFtEhsRjiO27vkBoZd/vn5d3b+XcrjFI9q\nxgXWkc8pUjOB93GMd3zgB9d1unjDSxmhWSHQqOJki+MtOuv/b3ZifUqctIrMxwhHuG1gim4Ma5kR\nWuEkXFgU4B0MW1Ho4TeyjX3s4TDq3zn8ALvZ7q5wX9DqLOHCDA95cFBAnOO1AL/ZdNopgY3fQcqF\nyriMe37hM9w521ZkkvlMo7o/8g7nZYQ/QDctp1nTCf0AAAAASUVORK5CYII=\n",zoom:"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAABmJLR0QA/wD/AP+gvaeTAAAACXBI\nWXMAAAsTAAALEwEAmpwYAAAAB3RJTUUH3gMPDiIRPL/2oQAAANBJREFUOMvF0b9KgzEcheHHVnCT\nKoI4uXbtLXgB3oJDJxevw1VwkoJ/NjepQ2/BrZRCx0ILFURQKV2kyOeSQpAmn7WDB0Lg955zEhLy\n2scdXlBggits+4WOQqjAJ3qYR7NGLrwXGU9+sGbEtlIF18FwmuBngZ+nCt6CIacC3Rx8LSl4xzgF\nn0tusBn4UyVhuA/7ZYIv5g+pE3ail25hN/qdmzCfpsJVjKKCZesDBwtzrAqGOMQj6vhCDRsY4ALH\nmOVObltR/xeG/jph6OD2r+Fv5lZBWEhMx58AAAAASUVORK5CYII=\n",brush:"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAABmJLR0QA/wD/AP+gvaeTAAAACXBI\nWXMAAEQkAABEJAFAZ8RUAAAAB3RJTUUH3gMCEiQKB9YaAgAAAWtJREFUOMuN0r1qVVEQhuFn700k\nnfEvBq0iNiIiOKXgH4KCaBeIhWARK/EibLwFCwVLjyAWaQzRGG9grC3URkHUBKKgRuWohWvL5pjj\nyTSLxcz7rZlZHyMiItqzFxGTEVF18/UoODNFxDIO4x12dkXqTcBPsCUzD+AK3ndFqhHwEsYz82gn\nN4dbmMRK9R/4KY7jAvbiWmYeHBT5Z4QCP8J1rGAeN3GvU3Mbl/Gq3qCDcxjLzOV+v78fq/iFIxFx\nPyJ2lNJpfBy2g59YzMyzEbEVLzGBJjOriLiBq5gaJrCIU3hcRCbwAtuwjm/Yg/V6I9NgDA1OR8RC\nZq6Vcd7iUwtn5h8fdMBdETGPE+Xe4ExELDRNs4bX2NfCUHe+7UExyfkCP8MhzOA7PuAkvrbwXyNF\nxF3MDqxiqlhXC7SPdaOKiN14g0u4g3H0MvOiTUSNY3iemb0ywmfMdfYyUmAJ2yPiBx6Wr/oy2Oqw\n+A1SupBzAOuE/AAAAABJRU5ErkJggg==\n"},T.Grid=h,h.prototype=Object.create(F.prototype),h.prototype.constructor=h,h.prototype.requiredProps=["xy"],h.prototype.defaultProps={color:"gray",dasharray:"2,2",alpha:"0.5",nticks:10,gridOn:!0,tickvalues:null,zorder:0},h.prototype.draw=function(){var s={left:"axisLeft",right:"axisRight",top:"axisTop",bottom:"axisBottom"}[this.position];this.grid=t[s](this.scale).ticks(this.props.nticks).tickValues(this.props.tickvalues).tickSize(this.tickSize,0,0).tickFormat(""),this.elem=this.ax.axes.append("g").attr("class",this.cssclass).attr("transform",this.transform).call(this.grid),T.insert_css("div#"+this.ax.fig.figid+" ."+this.cssclass+" .tick",{stroke:this.props.color,"stroke-dasharray":this.props.dasharray,"stroke-opacity":this.props.alpha}),T.insert_css("div#"+this.ax.fig.figid+" ."+this.cssclass+" path",{"stroke-width":0}),T.insert_css("div#"+this.ax.fig.figid+" ."+this.cssclass+" .domain",{"pointer-events":"none"})},h.prototype.zoomed=function(t){t?"x"==this.props.xy?this.elem.call(this.grid.scale(t.rescaleX(this.scale))):this.elem.call(this.grid.scale(t.rescaleY(this.scale))):this.elem.call(this.grid)},T.Axis=l,l.prototype=Object.create(F.prototype),l.prototype.constructor=l,l.prototype.requiredProps=["position"],l.prototype.defaultProps={nticks:10,tickvalues:null,tickformat:null,tickformat_formatter:null,fontsize:"11px",fontcolor:"black",axiscolor:"black",scale:"linear",grid:{},zorder:0,visible:!0},l.prototype.getGrid=function(){var t={nticks:this.props.nticks,zorder:this.props.zorder,tickvalues:null,xy:this.props.xy};if(this.props.grid)for(var s in this.props.grid)t[s]=this.props.grid[s];return new h(this.ax,t)},l.prototype.wrapTicks=function(){function s(s,i,e){e=e||1.2,s.each(function(){for(var s,o=t.select(this),r=o.node().getBBox(),n=r.height,a=o.text().split(/\s+/).reverse(),p=[],h=0,l=o.attr("y"),c=n,u=o.text(null).append("tspan").attr("x",0).attr("y",l).attr("dy",c);s=a.pop();)p.push(s),u.text(p.join(" ")),u.node().getComputedTextLength()>i&&(p.pop(),u.text(p.join(" ")),p=[s],u=o.append("tspan").attr("x",0).attr("y",l).attr("dy",++h*(n*e)+c).text(s))})}var i=80;"x"==this.props.xy&&this.elem.selectAll("text").call(s,i)},l.prototype.draw=function(){var s="x"===this.props.xy?this.parent.props.xscale:this.parent.props.yscale;if("date"===s&&this.props.tickvalues){var i="x"===this.props.xy?this.parent.x.domain():this.parent.y.domain(),e="x"===this.props.xy?this.parent.xdom.domain():this.parent.ydom.domain(),o=t.scaleLinear().domain(i).range(e);this.props.tickvalues=this.props.tickvalues.map(function(t){return new Date(o(t))})}var r={left:"axisLeft",right:"axisRight",top:"axisTop",bottom:"axisBottom"}[this.props.position];this.axis=t[r](this.scale);var n=this;"index"==this.props.tickformat_formatter?this.axis=this.axis.tickFormat(function(t,s){return n.props.tickformat[t]}):"percent"==this.props.tickformat_formatter?this.axis=this.axis.tickFormat(function(s,i){var e=s/n.props.tickformat.xmax*100,o=n.props.tickformat.decimals||0,r=t.format("."+o+"f")(e);return r+n.props.tickformat.symbol}):"str_method"==this.props.tickformat_formatter?this.axis=this.axis.tickFormat(function(s,i){var e=t.format(n.props.tickformat.format_string)(s);return n.props.tickformat.prefix+e+n.props.tickformat.suffix}):"fixed"==this.props.tickformat_formatter?this.axis=this.axis.tickFormat(function(t,s){return n.props.tickformat[s]}):this.tickFormat&&(this.axis=this.axis.tickFormat(this.tickFormat)),this.tickNr&&(this.axis=this.axis.ticks(this.tickNr)),this.props.tickvalues&&(this.axis=this.axis.tickValues(this.props.tickvalues),this.filter_ticks(this.axis.tickValues,this.axis.scale().domain())),this.elem=this.ax.baseaxes.append("g").attr("transform",this.transform).attr("class",this.cssclass).call(this.axis),this.wrapTicks(),T.insert_css("div#"+this.ax.fig.figid+" ."+this.cssclass+" line,  ."+this.cssclass+" path",{"shape-rendering":"crispEdges",stroke:this.props.axiscolor,fill:"none"}),T.insert_css("div#"+this.ax.fig.figid+" ."+this.cssclass+" text",{"font-family":"sans-serif","font-size":this.props.fontsize+"px",fill:this.props.fontcolor,stroke:"none"})},l.prototype.zoomed=function(t){this.props.tickvalues&&this.filter_ticks(this.axis.tickValues,this.axis.scale().domain()),t?("x"==this.props.xy?this.elem.call(this.axis.scale(t.rescaleX(this.scale))):this.elem.call(this.axis.scale(t.rescaleY(this.scale))),this.wrapTicks()):this.elem.call(this.axis)},l.prototype.setTicks=function(t,s){this.tickNr=t,this.tickFormat=s},l.prototype.filter_ticks=function(t,s){null!=this.props.tickvalues&&t(this.props.tickvalues.filter(function(t){return t>=s[0]&&t<=s[1]}))},T.Coordinates=c,c.prototype.xy=function(t,s,i){return s="undefined"==typeof s?0:s,i="undefined"==typeof i?1:i,[this.x(t[s]),this.y(t[i])]},c.prototype.x_data=function(t){return this.ax.x(t)},c.prototype.y_data=function(t){return this.ax.y(t)},c.prototype.x_display=function(t){return t},c.prototype.y_display=function(t){return t},c.prototype.x_axes=function(t){return t*this.ax.width},c.prototype.y_axes=function(t){return this.ax.height*(1-t)},c.prototype.x_figure=function(t){return t*this.fig.width-this.ax.position[0]},c.prototype.y_figure=function(t){return(1-t)*this.fig.height-this.ax.position[1]},T.Path=u,u.prototype=Object.create(F.prototype),u.prototype.constructor=u,u.prototype.requiredProps=["data"],u.prototype.defaultProps={xindex:0,yindex:1,coordinates:"data",facecolor:"green",edgecolor:"black",edgewidth:1,dasharray:"none",pathcodes:null,offset:null,offsetcoordinates:"data",alpha:1,drawstyle:"none",zorder:1},u.prototype.finiteFilter=function(t,s){return isFinite(this.pathcoords.x(t[this.props.xindex]))&&isFinite(this.pathcoords.y(t[this.props.yindex]))},u.prototype.draw=function(){if(this.datafunc.defined(this.finiteFilter.bind(this)).x(function(t){return this.pathcoords.x(t[this.props.xindex])}.bind(this)).y(function(t){return this.pathcoords.y(t[this.props.yindex])}.bind(this)),this.pathcoords.zoomable?this.path=this.ax.paths.append("svg:path"):this.path=this.ax.staticPaths.append("svg:path"),this.path=this.path.attr("d",this.datafunc(this.data,this.pathcodes)).attr("class","mpld3-path").style("stroke",this.props.edgecolor).style("stroke-width",this.props.edgewidth).style("stroke-dasharray",this.props.dasharray).style("stroke-opacity",this.props.alpha).style("fill",this.props.facecolor).style("fill-opacity",this.props.alpha).attr("vector-effect","non-scaling-stroke"),null!==this.props.offset){var t=this.offsetcoords.xy(this.props.offset);this.path.attr("transform","translate("+t+")")}},u.prototype.elements=function(t){return this.path},T.PathCollection=d,d.prototype=Object.create(F.prototype),d.prototype.constructor=d,d.prototype.requiredProps=["paths","offsets"],d.prototype.defaultProps={xindex:0,yindex:1,pathtransforms:[],pathcoordinates:"display",offsetcoordinates:"data",offsetorder:"before",edgecolors:["#000000"],drawstyle:"none",edgewidths:[1],facecolors:["#0000FF"],alphas:[1],zorder:2},d.prototype.transformFunc=function(t,s){var i=this.props.pathtransforms,e=0==i.length?"":T.getTransformation("matrix("+n(i,s)+")").toString(),o=null===t||"undefined"==typeof t?"translate(0, 0)":"translate("+this.offsetcoords.xy(t,this.props.xindex,this.props.yindex)+")";return"after"===this.props.offsetorder?e+o:o+e},d.prototype.pathFunc=function(t,s){return a().x(function(t){return this.pathcoords.x(t[0])}.bind(this)).y(function(t){return this.pathcoords.y(t[1])}.bind(this)).apply(this,n(this.props.paths,s))},d.prototype.styleFunc=function(t,s){var i={stroke:n(this.props.edgecolors,s),"stroke-width":n(this.props.edgewidths,s),"stroke-opacity":n(this.props.alphas,s),fill:n(this.props.facecolors,s),"fill-opacity":n(this.props.alphas,s)},e="";for(var o in i)e+=o+":"+i[o]+";";return e},d.prototype.allFinite=function(t){return t instanceof Array?t.length==t.filter(isFinite).length:!0},d.prototype.draw=function(){this.offsetcoords.zoomable||this.pathcoords.zoomable?this.group=this.ax.paths.append("svg:g"):this.group=this.ax.staticPaths.append("svg:g"),this.pathsobj=this.group.selectAll("paths").data(this.offsets.filter(this.allFinite)).enter().append("svg:path").attr("d",this.pathFunc.bind(this)).attr("class","mpld3-path").attr("transform",this.transformFunc.bind(this)).attr("style",this.styleFunc.bind(this)).attr("vector-effect","non-scaling-stroke")},d.prototype.elements=function(t){return this.group.selectAll("path")},T.Line=f,f.prototype=Object.create(u.prototype),f.prototype.constructor=f,f.prototype.requiredProps=["data"],f.prototype.defaultProps={xindex:0,yindex:1,coordinates:"data",color:"salmon",linewidth:2,dasharray:"none",alpha:1,zorder:2,drawstyle:"none"},T.Markers=y,y.prototype=Object.create(d.prototype),y.prototype.constructor=y,y.prototype.requiredProps=["data"],y.prototype.defaultProps={xindex:0,yindex:1,coordinates:"data",facecolor:"salmon",edgecolor:"black",edgewidth:1,alpha:1,markersize:6,markername:"circle",drawstyle:"none",markerpath:null,zorder:3},y.prototype.pathFunc=function(t,s){return this.marker},T.Image=m,m.prototype=Object.create(F.prototype),m.prototype.constructor=m,m.prototype.requiredProps=["data","extent"],m.prototype.defaultProps={alpha:1,coordinates:"data",drawstyle:"none",zorder:1},m.prototype.draw=function(){this.image=this.ax.paths.append("svg:image"),this.image=this.image.attr("class","mpld3-image").attr("xlink:href","data:image/png;base64,"+this.props.data).style("opacity",this.props.alpha).attr("preserveAspectRatio","none"),this.updateDimensions()},m.prototype.elements=function(s){return t.select(this.image)},m.prototype.updateDimensions=function(){var t=this.props.extent;this.image.attr("x",this.coords.x(t[0])).attr("y",this.coords.y(t[3])).attr("width",this.coords.x(t[1])-this.coords.x(t[0])).attr("height",this.coords.y(t[2])-this.coords.y(t[3]))},T.Text=g,g.prototype=Object.create(F.prototype),g.prototype.constructor=g,g.prototype.requiredProps=["text","position"],g.prototype.defaultProps={coordinates:"data",h_anchor:"start",v_baseline:"auto",rotation:0,fontsize:11,drawstyle:"none",color:"black",alpha:1,zorder:3},g.prototype.draw=function(){"data"==this.props.coordinates?this.coords.zoomable?this.obj=this.ax.paths.append("text"):this.obj=this.ax.staticPaths.append("text"):this.obj=this.ax.baseaxes.append("text"),this.obj.attr("class","mpld3-text").text(this.text).style("text-anchor",this.props.h_anchor).style("dominant-baseline",this.props.v_baseline).style("font-size",this.props.fontsize).style("fill",this.props.color).style("opacity",this.props.alpha),this.applyTransform()},g.prototype.elements=function(s){return t.select(this.obj)},g.prototype.applyTransform=function(){var t=this.coords.xy(this.position);this.obj.attr("x",t[0]).attr("y",t[1]),this.props.rotation&&this.obj.attr("transform","rotate("+this.props.rotation+","+t+")")},T.Axes=x,x.prototype=Object.create(F.prototype),x.prototype.constructor=x,x.prototype.requiredProps=["xlim","ylim"],x.prototype.defaultProps={bbox:[.1,.1,.8,.8],axesbg:"#FFFFFF",axesbgalpha:1,gridOn:!1,xdomain:null,ydomain:null,xscale:"linear",yscale:"linear",zoomable:!0,axes:[{position:"left"},{position:"bottom"}],lines:[],paths:[],markers:[],texts:[],collections:[],sharex:[],sharey:[],images:[]},x.prototype.draw=function(){for(var s=0;s<this.props.sharex.length;s++)this.sharex.push(T.get_element(this.props.sharex[s]));for(var s=0;s<this.props.sharey.length;s++)this.sharey.push(T.get_element(this.props.sharey[s]));this.baseaxes=this.fig.canvas.append("g").attr("transform","translate("+this.position[0]+","+this.position[1]+")").attr("width",this.width).attr("height",this.height).attr("class","mpld3-baseaxes"),this.axes=this.baseaxes.append("g").attr("class","mpld3-axes").style("pointer-events","visiblefill"),this.clip=this.axes.append("svg:clipPath").attr("id",this.clipid).append("svg:rect").attr("x",0).attr("y",0).attr("width",this.width).attr("height",this.height),this.axesbg=this.axes.append("svg:rect").attr("width",this.width).attr("height",this.height).attr("class","mpld3-axesbg").style("fill",this.props.axesbg).style("fill-opacity",this.props.axesbgalpha),this.pathsContainer=this.axes.append("g").attr("clip-path","url(#"+this.clipid+")").attr("x",0).attr("y",0).attr("width",this.width).attr("height",this.height).attr("class","mpld3-paths-container"),this.paths=this.pathsContainer.append("g").attr("class","mpld3-paths"),this.staticPaths=this.axes.append("g").attr("class","mpld3-staticpaths"),this.brush=t.brush().extent([[0,0],[this.fig.width,this.fig.height]]).on("start",this.brushStart.bind(this)).on("brush",this.brushMove.bind(this)).on("end",this.brushEnd.bind(this)).on("start.nokey",function(){t.select(window).on("keydown.brush keyup.brush",null)});for(var s=0;s<this.elements.length;s++)this.elements[s].draw()},x.prototype.bindZoom=function(){this.zoom||(this.zoom=t.zoom(),this.zoom.on("zoom",this.zoomed.bind(this)),this.axes.call(this.zoom))},x.prototype.unbindZoom=function(){this.zoom&&(this.zoom.on("zoom",null),this.axes.on(".zoom",null),this.zoom=null)},x.prototype.bindBrush=function(){this.brushG||(this.brushG=this.axes.append("g").attr("class","mpld3-brush").call(this.brush))},x.prototype.unbindBrush=function(){this.brushG&&(this.brushG.remove(),this.brushG.on(".brush",null),this.brushG=null)},x.prototype.reset=function(){this.zoom?this.doZoom(!1,t.zoomIdentity,750):(this.bindZoom(),this.doZoom(!1,t.zoomIdentity,750,function(){this.isSomeTypeOfZoomEnabled||this.unbindZoom()}.bind(this)))},x.prototype.enableOrDisableBrushing=function(){this.isBoxzoomEnabled||this.isLinkedBrushEnabled?this.bindBrush():this.unbindBrush()},x.prototype.isSomeTypeOfZoomEnabled=function(){return this.isZoomEnabled||this.isBoxzoomEnabled},x.prototype.enableOrDisableZooming=function(){this.isSomeTypeOfZoomEnabled()?this.bindZoom():this.unbindZoom()},x.prototype.enableLinkedBrush=function(){this.isLinkedBrushEnabled=!0,this.enableOrDisableBrushing()},x.prototype.disableLinkedBrush=function(){this.isLinkedBrushEnabled=!1,this.enableOrDisableBrushing()},x.prototype.enableBoxzoom=function(){this.isBoxzoomEnabled=!0,this.enableOrDisableBrushing(),this.enableOrDisableZooming()},x.prototype.disableBoxzoom=function(){this.isBoxzoomEnabled=!1,this.enableOrDisableBrushing(),this.enableOrDisableZooming()},x.prototype.enableZoom=function(){this.isZoomEnabled=!0,this.enableOrDisableZooming(),this.axes.style("cursor","move")},x.prototype.disableZoom=function(){this.isZoomEnabled=!1,this.enableOrDisableZooming(),this.axes.style("cursor",null)},x.prototype.doZoom=function(t,s,i,e){if(this.props.zoomable&&this.zoom){if(i){var o=this.axes.transition().duration(i).call(this.zoom.transform,s);e&&o.on("end",e)}else this.axes.call(this.zoom.transform,s);t?(this.lastTransform=s,this.sharex.forEach(function(t){t.doZoom(!1,s,i)}),this.sharey.forEach(function(t){t.doZoom(!1,s,i)})):this.lastTransform=s}},x.prototype.zoomed=function(){var s=t.event.sourceEvent&&"zoom"!=t.event.sourceEvent.type;if(s)this.doZoom(!0,t.event.transform,!1);else{var i=t.event.transform;this.paths.attr("transform",i),this.elements.forEach(function(t){t.zoomed&&t.zoomed(i)}.bind(this))}},x.prototype.resetBrush=function(){this.brushG.call(this.brush.move,null)},x.prototype.doBoxzoom=function(s){if(s&&this.brushG){var i=s.map(this.lastTransform.invert,this.lastTransform),e=i[1][0]-i[0][0],o=i[1][1]-i[0][1],r=(i[0][0]+i[1][0])/2,n=(i[0][1]+i[1][1])/2,a=e>o?this.width/e:this.height/o,p=this.width/2-a*r,h=this.height/2-a*n,l=t.zoomIdentity.translate(p,h).scale(a);this.doZoom(!0,l,750),this.resetBrush()}},x.prototype.brushStart=function(){this.isLinkedBrushEnabled&&(this.isCurrentLinkedBrushTarget="MouseEvent"==t.event.sourceEvent.constructor.name,this.isCurrentLinkedBrushTarget&&this.fig.resetBrushForOtherAxes(this.axid))},x.prototype.brushMove=function(){var s=t.event.selection;this.isLinkedBrushEnabled&&this.fig.updateLinkedBrush(s)},x.prototype.brushEnd=function(){var s=t.event.selection;this.isBoxzoomEnabled&&this.doBoxzoom(s),this.isLinkedBrushEnabled&&(s||this.fig.endLinkedBrush(),this.isCurrentLinkedBrushTarget=!1)},x.prototype.setTicks=function(t,s,i){this.axisList.forEach(function(e){e.props.xy==t&&e.setTicks(s,i)})},T.Toolbar=b,b.prototype=Object.create(F.prototype),b.prototype.constructor=b,b.prototype.defaultProps={buttons:["reset","move"]},b.prototype.addButton=function(t){this.buttons.push(new t(this))},b.prototype.draw=function(){function s(){this.buttonsobj.transition(750).attr("y",0)}function i(){this.buttonsobj.transition(750).delay(250).attr("y",16)}T.insert_css("div#"+this.fig.figid+" .mpld3-toolbar image",{cursor:"pointer",opacity:.2,display:"inline-block",margin:"0px"}),T.insert_css("div#"+this.fig.figid+" .mpld3-toolbar image.active",{opacity:.4}),T.insert_css("div#"+this.fig.figid+" .mpld3-toolbar image.pressed",{opacity:.6}),this.fig.canvas.on("mouseenter",s.bind(this)).on("mouseleave",i.bind(this)).on("touchenter",s.bind(this)).on("touchstart",s.bind(this)),this.toolbar=this.fig.canvas.append("svg:svg").attr("width",16*this.buttons.length).attr("height",16).attr("x",2).attr("y",this.fig.height-16-2).attr("class","mpld3-toolbar"),this.buttonsobj=this.toolbar.append("svg:g").selectAll("buttons").data(this.buttons).enter().append("svg:image").attr("class",function(t){
return t.cssclass}).attr("xlink:href",function(t){return t.icon()}).attr("width",16).attr("height",16).attr("x",function(t,s){return 16*s}).attr("y",16).on("click",function(t){t.click()}).on("mouseenter",function(){t.select(this).classed("active",!0)}).on("mouseleave",function(){t.select(this).classed("active",!1)});for(var e=0;e<this.buttons.length;e++)this.buttons[e].onDraw()},b.prototype.deactivate_all=function(){this.buttons.forEach(function(t){t.deactivate()})},b.prototype.deactivate_by_action=function(t){function s(s){return-1!==t.indexOf(s)}t.length>0&&this.buttons.forEach(function(t){t.actions.filter(s).length>0&&t.deactivate()})},T.Button=v,v.prototype=Object.create(F.prototype),v.prototype.constructor=v,v.prototype.setState=function(t){t?this.activate():this.deactivate()},v.prototype.click=function(){this.active?this.deactivate():this.activate()},v.prototype.activate=function(){this.toolbar.deactivate_by_action(this.actions),this.onActivate(),this.active=!0,this.toolbar.toolbar.select("."+this.cssclass).classed("pressed",!0),this.sticky||this.deactivate()},v.prototype.deactivate=function(){this.onDeactivate(),this.active=!1,this.toolbar.toolbar.select("."+this.cssclass).classed("pressed",!1)},v.prototype.sticky=!1,v.prototype.actions=[],v.prototype.icon=function(){return""},v.prototype.onActivate=function(){},v.prototype.onDeactivate=function(){},v.prototype.onDraw=function(){},T.ButtonFactory=function(t){function s(t){v.call(this,t,this.buttonID)}if("string"!=typeof t.buttonID)throw"ButtonFactory: buttonID must be present and be a string";s.prototype=Object.create(v.prototype),s.prototype.constructor=s;for(var i in t)s.prototype[i]=t[i];return s},T.Plugin=A,A.prototype=Object.create(F.prototype),A.prototype.constructor=A,A.prototype.requiredProps=[],A.prototype.defaultProps={},A.prototype.draw=function(){},T.ResetPlugin=k,T.register_plugin("reset",k),k.prototype=Object.create(A.prototype),k.prototype.constructor=k,k.prototype.requiredProps=[],k.prototype.defaultProps={},T.ZoomPlugin=w,T.register_plugin("zoom",w),w.prototype=Object.create(A.prototype),w.prototype.constructor=w,w.prototype.requiredProps=[],w.prototype.defaultProps={button:!0,enabled:null},w.prototype.activate=function(){this.fig.enableZoom()},w.prototype.deactivate=function(){this.fig.disableZoom()},w.prototype.draw=function(){this.props.enabled?this.activate():this.deactivate()},T.BoxZoomPlugin=B,T.register_plugin("boxzoom",B),B.prototype=Object.create(A.prototype),B.prototype.constructor=B,B.prototype.requiredProps=[],B.prototype.defaultProps={button:!0,enabled:null},B.prototype.activate=function(){this.fig.enableBoxzoom()},B.prototype.deactivate=function(){this.fig.disableBoxzoom()},B.prototype.draw=function(){this.props.enabled?this.activate():this.deactivate()},T.TooltipPlugin=z,T.register_plugin("tooltip",z),z.prototype=Object.create(A.prototype),z.prototype.constructor=z,z.prototype.requiredProps=["id"],z.prototype.defaultProps={labels:null,hoffset:0,voffset:10,location:"mouse"},z.prototype.draw=function(){function s(t,s){this.tooltip.style("visibility","visible").text(null===r?"("+t+")":n(r,s))}function i(s,i){if("mouse"===a){var e=t.mouse(this.fig.canvas.node());this.x=e[0]+this.props.hoffset,this.y=e[1]-this.props.voffset}this.tooltip.attr("x",this.x).attr("y",this.y)}function e(t,s){this.tooltip.style("visibility","hidden")}var o=T.get_element(this.props.id,this.fig),r=this.props.labels,a=this.props.location;this.tooltip=this.fig.canvas.append("text").attr("class","mpld3-tooltip-text").attr("x",0).attr("y",0).text("").style("visibility","hidden"),"bottom left"==a||"top left"==a?(this.x=o.ax.position[0]+5+this.props.hoffset,this.tooltip.style("text-anchor","beginning")):"bottom right"==a||"top right"==a?(this.x=o.ax.position[0]+o.ax.width-5+this.props.hoffset,this.tooltip.style("text-anchor","end")):this.tooltip.style("text-anchor","middle"),"bottom left"==a||"bottom right"==a?this.y=o.ax.position[1]+o.ax.height-5+this.props.voffset:("top left"==a||"top right"==a)&&(this.y=o.ax.position[1]+5+this.props.voffset),o.elements().on("mouseover",s.bind(this)).on("mousemove",i.bind(this)).on("mouseout",e.bind(this))},T.LinkedBrushPlugin=E,T.register_plugin("linkedbrush",E),E.prototype=Object.create(T.Plugin.prototype),E.prototype.constructor=E,E.prototype.requiredProps=["id"],E.prototype.defaultProps={button:!0,enabled:null},E.prototype.activate=function(){this.fig.enableLinkedBrush()},E.prototype.deactivate=function(){this.fig.disableLinkedBrush()},E.prototype.isPathInSelection=function(t,s,i,e){var o=e[0][0]<t[s]&&e[1][0]>t[s]&&e[0][1]<t[i]&&e[1][1]>t[i];return o},E.prototype.invertSelection=function(t,s){var i=[s.x.invert(t[0][0]),s.x.invert(t[1][0])],e=[s.y.invert(t[1][1]),s.y.invert(t[0][1])];return[[Math.min.apply(Math,i),Math.min.apply(Math,e)],[Math.max.apply(Math,i),Math.max.apply(Math,e)]]},E.prototype.update=function(t){t&&this.pathCollectionsByAxes.forEach(function(s,i){var e=s[0],o=this.objectsByAxes[i],r=this.invertSelection(t,this.fig.axes[i]),n=e.props.xindex,a=e.props.yindex;o.selectAll("path").classed("mpld3-hidden",function(t,s){return!this.isPathInSelection(t,n,a,r)}.bind(this))}.bind(this))},E.prototype.end=function(){this.allObjects.selectAll("path").classed("mpld3-hidden",!1)},E.prototype.draw=function(){T.insert_css("#"+this.fig.figid+" path.mpld3-hidden",{stroke:"#ccc !important",fill:"#ccc !important"});var t=T.get_element(this.props.id);if(!t)throw new Error("[LinkedBrush] Could not find path collection");if(!("offsets"in t.props))throw new Error("[LinkedBrush] Figure is not a scatter plot.");this.objectClass="mpld3-brushtarget-"+t.props[this.dataKey],this.pathCollectionsByAxes=this.fig.axes.map(function(s){return s.elements.map(function(s){return s.props[this.dataKey]==t.props[this.dataKey]?(s.group.classed(this.objectClass,!0),s):void 0}.bind(this)).filter(function(t){return t})}.bind(this)),this.objectsByAxes=this.fig.axes.map(function(t){return t.axes.selectAll("."+this.objectClass)}.bind(this)),this.allObjects=this.fig.canvas.selectAll("."+this.objectClass)},T.register_plugin("mouseposition",P),P.prototype=Object.create(T.Plugin.prototype),P.prototype.constructor=P,P.prototype.requiredProps=[],P.prototype.defaultProps={fontsize:12,fmt:".3g"},P.prototype.draw=function(){for(var s=this.fig,i=t.format(this.props.fmt),e=s.canvas.append("text").attr("class","mpld3-coordinates").style("text-anchor","end").style("font-size",this.props.fontsize).attr("x",this.fig.width-5).attr("y",this.fig.height-5),o=0;o<this.fig.axes.length;o++){var r=function(){var r=s.axes[o];return function(){var s=t.mouse(this),o=r.x.invert(s[0]),n=r.y.invert(s[1]);e.text("("+i(o)+", "+i(n)+")")}}();s.axes[o].baseaxes.on("mousemove",r).on("mouseout",function(){e.text("")})}},T.Figure=O,O.prototype=Object.create(F.prototype),O.prototype.constructor=O,O.prototype.requiredProps=["width","height"],O.prototype.defaultProps={data:{},axes:[],plugins:[{type:"reset"},{type:"zoom"},{type:"boxzoom"}]},O.prototype.addPlugin=function(t){if(!t.type)return console.warn("unspecified plugin type. Skipping this");var i;if(!(t.type in T.plugin_map))return console.warn("Skipping unrecognized plugin: "+i);i=T.plugin_map[t.type],(t.clear_toolbar||t.buttons)&&console.warn("DEPRECATION WARNING: You are using pluginInfo.clear_toolbar or pluginInfo, which have been deprecated. Please see the build-in plugins for the new method to add buttons, otherwise contact the mpld3 maintainers.");var e=s(t);delete e.type;var o=new i(this,e);this.plugins.push(o),this.pluginsByType[t.type]=o},O.prototype.draw=function(){T.insert_css("div#"+this.figid,{"font-family":"Helvetica, sans-serif"}),this.canvas=this.root.append("svg:svg").attr("class","mpld3-figure").attr("width",this.width).attr("height",this.height);for(var t=0;t<this.axes.length;t++)this.axes[t].draw();this.disableZoom();for(var t=0;t<this.plugins.length;t++)this.plugins[t].draw();this.toolbar.draw()},O.prototype.resetBrushForOtherAxes=function(t){this.axes.forEach(function(s){s.axid!=t&&s.resetBrush()})},O.prototype.updateLinkedBrush=function(t){this.pluginsByType.linkedbrush&&this.pluginsByType.linkedbrush.update(t)},O.prototype.endLinkedBrush=function(){this.pluginsByType.linkedbrush&&this.pluginsByType.linkedbrush.end()},O.prototype.reset=function(t){this.axes.forEach(function(t){t.reset()})},O.prototype.enableLinkedBrush=function(){this.axes.forEach(function(t){t.enableLinkedBrush()})},O.prototype.disableLinkedBrush=function(){this.axes.forEach(function(t){t.disableLinkedBrush()})},O.prototype.enableBoxzoom=function(){this.axes.forEach(function(t){t.enableBoxzoom()})},O.prototype.disableBoxzoom=function(){this.axes.forEach(function(t){t.disableBoxzoom()})},O.prototype.enableZoom=function(){this.axes.forEach(function(t){t.enableZoom()})},O.prototype.disableZoom=function(){this.axes.forEach(function(t){t.disableZoom()})},O.prototype.toggleZoom=function(){this.isZoomEnabled?this.disableZoom():this.enableZoom()},O.prototype.setTicks=function(t,s,i){this.axes.forEach(function(e){e.setTicks(t,s,i)})},O.prototype.setXTicks=function(t,s){this.setTicks("x",t,s)},O.prototype.setYTicks=function(t,s){this.setTicks("y",t,s)},O.prototype.removeNaN=function(t){output=output.map(function(t){return t.map(function(t){return"number"==typeof t&&isNaN(t)?0:t})})},O.prototype.parse_offsets=function(t){return t.map(function(t){return t.map(function(t){return"number"==typeof t&&isNaN(t)?0:t})})},O.prototype.get_data=function(t){var s=t;return null===t||"undefined"==typeof t?s=null:"string"==typeof t&&(s=this.data[t]),s},T.PlotElement=F,F.prototype.requiredProps=[],F.prototype.defaultProps={},F.prototype.processProps=function(t){t=s(t);var i={},e=this.name();this.requiredProps.forEach(function(s){if(!(s in t))throw"property '"+s+"' must be specified for "+e;i[s]=t[s],delete t[s]});for(var o in this.defaultProps)o in t?(i[o]=t[o],delete t[o]):i[o]=this.defaultProps[o];"id"in t?(i.id=t.id,delete t.id):"id"in i||(i.id=T.generateId());for(var o in t)console.warn("Unrecognized property '"+o+"' for object "+this.name()+" (value = "+t[o]+").");return i},F.prototype.name=function(){var t=/function (.{1,})\(/,s=t.exec(this.constructor.toString());return s&&s.length>1?s[1]:""},"object"==typeof module&&module.exports?module.exports=T:this.mpld3=T,console.log("Loaded mpld3 version "+T.version)}(d3);
},{}]},{},[1])(1)
});

}).call(this,typeof global !== "undefined" ? global : typeof self !== "undefined" ? self : typeof window !== "undefined" ? window : {})
},{}],4:[function(require,module,exports){
// shim for using process in browser
var process = module.exports = {};

// cached from whatever global is present so that test runners that stub it
// don't break things.  But we need to wrap it in a try catch in case it is
// wrapped in strict mode code which doesn't define any globals.  It's inside a
// function because try/catches deoptimize in certain engines.

var cachedSetTimeout;
var cachedClearTimeout;

function defaultSetTimout() {
    throw new Error('setTimeout has not been defined');
}
function defaultClearTimeout () {
    throw new Error('clearTimeout has not been defined');
}
(function () {
    try {
        if (typeof setTimeout === 'function') {
            cachedSetTimeout = setTimeout;
        } else {
            cachedSetTimeout = defaultSetTimout;
        }
    } catch (e) {
        cachedSetTimeout = defaultSetTimout;
    }
    try {
        if (typeof clearTimeout === 'function') {
            cachedClearTimeout = clearTimeout;
        } else {
            cachedClearTimeout = defaultClearTimeout;
        }
    } catch (e) {
        cachedClearTimeout = defaultClearTimeout;
    }
} ())
function runTimeout(fun) {
    if (cachedSetTimeout === setTimeout) {
        //normal enviroments in sane situations
        return setTimeout(fun, 0);
    }
    // if setTimeout wasn't available but was latter defined
    if ((cachedSetTimeout === defaultSetTimout || !cachedSetTimeout) && setTimeout) {
        cachedSetTimeout = setTimeout;
        return setTimeout(fun, 0);
    }
    try {
        // when when somebody has screwed with setTimeout but no I.E. maddness
        return cachedSetTimeout(fun, 0);
    } catch(e){
        try {
            // When we are in I.E. but the script has been evaled so I.E. doesn't trust the global object when called normally
            return cachedSetTimeout.call(null, fun, 0);
        } catch(e){
            // same as above but when it's a version of I.E. that must have the global object for 'this', hopfully our context correct otherwise it will throw a global error
            return cachedSetTimeout.call(this, fun, 0);
        }
    }


}
function runClearTimeout(marker) {
    if (cachedClearTimeout === clearTimeout) {
        //normal enviroments in sane situations
        return clearTimeout(marker);
    }
    // if clearTimeout wasn't available but was latter defined
    if ((cachedClearTimeout === defaultClearTimeout || !cachedClearTimeout) && clearTimeout) {
        cachedClearTimeout = clearTimeout;
        return clearTimeout(marker);
    }
    try {
        // when when somebody has screwed with setTimeout but no I.E. maddness
        return cachedClearTimeout(marker);
    } catch (e){
        try {
            // When we are in I.E. but the script has been evaled so I.E. doesn't  trust the global object when called normally
            return cachedClearTimeout.call(null, marker);
        } catch (e){
            // same as above but when it's a version of I.E. that must have the global object for 'this', hopfully our context correct otherwise it will throw a global error.
            // Some versions of I.E. have different rules for clearTimeout vs setTimeout
            return cachedClearTimeout.call(this, marker);
        }
    }



}
var queue = [];
var draining = false;
var currentQueue;
var queueIndex = -1;

function cleanUpNextTick() {
    if (!draining || !currentQueue) {
        return;
    }
    draining = false;
    if (currentQueue.length) {
        queue = currentQueue.concat(queue);
    } else {
        queueIndex = -1;
    }
    if (queue.length) {
        drainQueue();
    }
}

function drainQueue() {
    if (draining) {
        return;
    }
    var timeout = runTimeout(cleanUpNextTick);
    draining = true;

    var len = queue.length;
    while(len) {
        currentQueue = queue;
        queue = [];
        while (++queueIndex < len) {
            if (currentQueue) {
                currentQueue[queueIndex].run();
            }
        }
        queueIndex = -1;
        len = queue.length;
    }
    currentQueue = null;
    draining = false;
    runClearTimeout(timeout);
}

process.nextTick = function (fun) {
    var args = new Array(arguments.length - 1);
    if (arguments.length > 1) {
        for (var i = 1; i < arguments.length; i++) {
            args[i - 1] = arguments[i];
        }
    }
    queue.push(new Item(fun, args));
    if (queue.length === 1 && !draining) {
        runTimeout(drainQueue);
    }
};

// v8 likes predictible objects
function Item(fun, array) {
    this.fun = fun;
    this.array = array;
}
Item.prototype.run = function () {
    this.fun.apply(null, this.array);
};
process.title = 'browser';
process.browser = true;
process.env = {};
process.argv = [];
process.version = ''; // empty string to avoid regexp issues
process.versions = {};

function noop() {}

process.on = noop;
process.addListener = noop;
process.once = noop;
process.off = noop;
process.removeListener = noop;
process.removeAllListeners = noop;
process.emit = noop;
process.prependListener = noop;
process.prependOnceListener = noop;

process.listeners = function (name) { return [] }

process.binding = function (name) {
    throw new Error('process.binding is not supported');
};

process.cwd = function () { return '/' };
process.chdir = function (dir) {
    throw new Error('process.chdir is not supported');
};
process.umask = function() { return 0; };

},{}],5:[function(require,module,exports){
(function (setImmediate,clearImmediate){
var nextTick = require('process/browser.js').nextTick;
var apply = Function.prototype.apply;
var slice = Array.prototype.slice;
var immediateIds = {};
var nextImmediateId = 0;

// DOM APIs, for completeness

exports.setTimeout = function() {
  return new Timeout(apply.call(setTimeout, window, arguments), clearTimeout);
};
exports.setInterval = function() {
  return new Timeout(apply.call(setInterval, window, arguments), clearInterval);
};
exports.clearTimeout =
exports.clearInterval = function(timeout) { timeout.close(); };

function Timeout(id, clearFn) {
  this._id = id;
  this._clearFn = clearFn;
}
Timeout.prototype.unref = Timeout.prototype.ref = function() {};
Timeout.prototype.close = function() {
  this._clearFn.call(window, this._id);
};

// Does not start the time, just sets up the members needed.
exports.enroll = function(item, msecs) {
  clearTimeout(item._idleTimeoutId);
  item._idleTimeout = msecs;
};

exports.unenroll = function(item) {
  clearTimeout(item._idleTimeoutId);
  item._idleTimeout = -1;
};

exports._unrefActive = exports.active = function(item) {
  clearTimeout(item._idleTimeoutId);

  var msecs = item._idleTimeout;
  if (msecs >= 0) {
    item._idleTimeoutId = setTimeout(function onTimeout() {
      if (item._onTimeout)
        item._onTimeout();
    }, msecs);
  }
};

// That's not how node.js implements it but the exposed api is the same.
exports.setImmediate = typeof setImmediate === "function" ? setImmediate : function(fn) {
  var id = nextImmediateId++;
  var args = arguments.length < 2 ? false : slice.call(arguments, 1);

  immediateIds[id] = true;

  nextTick(function onNextTick() {
    if (immediateIds[id]) {
      // fn.call() is faster so we optimize for the common use-case
      // @see http://jsperf.com/call-apply-segu
      if (args) {
        fn.apply(null, args);
      } else {
        fn.call(null);
      }
      // Prevent ids from leaking
      exports.clearImmediate(id);
    }
  });

  return id;
};

exports.clearImmediate = typeof clearImmediate === "function" ? clearImmediate : function(id) {
  delete immediateIds[id];
};
}).call(this,require("timers").setImmediate,require("timers").clearImmediate)
},{"process/browser.js":4,"timers":5}]},{},[2])(2)
});
