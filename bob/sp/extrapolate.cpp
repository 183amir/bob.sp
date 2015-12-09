/**
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Thu 30 Jan 17:13:00 2014 CET
 *
 * @brief Binds extrapolation to python
 *
 * Copyright (C) 2011-2014 Idiap Research Institute, Martigny, Switzerland
 */

#include "main.h"

static auto s_bordertype = bob::extension::ClassDoc(
  BOB_EXT_MODULE_PREFIX ".BorderType",
  "An enumeration to define different types of border handling in extrapolations (and other functions)",
  "This class is just a representation of the BorderType C++ ``enum`` for Python. "
  "This class cannot be instantiated in Python. "
  "Instead, Use of the values available in this class as input for ``BorderType`` when required:\n\n"
  "* Zero : Fills the border with 0 (a special version of ``Constant``)\n"
  "* Constant : Fills the border with a given constant value\n"
  "* NearestNeighbour : Fills the border with the nearest neighbor from the inside\n"
  "* Circular : Fills the border by copying data from the other side of the array\n"
  "* Mirror : Fills the border by copying data from the inside in a mirroring way\n\n"
  "A dictionary containing all names and values available for this enumeration is available through :py:attr:`entries`."
);

PyTypeObject PyBobSpExtrapolationBorder_Type = {
    PyVarObject_HEAD_INIT(0, 0)
    0
};

static int insert(PyObject* dict, PyObject* entries, const char* key, Py_ssize_t value) {
  // inserts the item to both the dictionary and the entries dictionary
  auto v = make_safe(Py_BuildValue("n", value));
  if (PyDict_SetItemString(dict, key, v.get()) < 0) return -1;
  return PyDict_SetItemString(entries, key, v.get());
}

static PyObject* create_enumerations() {
  auto retval = PyDict_New();
  if (!retval) return 0;
  auto retval_ = make_safe(retval);

  auto entries = PyDict_New();
  if (!entries) return 0;
  auto entries_ = make_safe(entries);

  if (insert(retval, entries, "Zero", bob::sp::Extrapolation::BorderType::Zero) < 0) return 0;
  if (insert(retval, entries, "Constant", bob::sp::Extrapolation::BorderType::Constant) < 0) return 0;
  if (insert(retval, entries, "NearestNeighbour", bob::sp::Extrapolation::BorderType::NearestNeighbour) < 0) return 0;
  if (insert(retval, entries, "Circular", bob::sp::Extrapolation::BorderType::Circular) < 0) return 0;
  if (insert(retval, entries, "Mirror", bob::sp::Extrapolation::BorderType::Mirror) < 0) return 0;

  if (PyDict_SetItemString(retval, "entries", entries) < 0) return 0;

  return Py_BuildValue("O", retval);
}

int PyBobSpExtrapolationBorder_Converter(PyObject* o, bob::sp::Extrapolation::BorderType* b) {
  if (!o) return 0;

  if (PyInt_Check(o)){
    Py_ssize_t v = PyNumber_AsSsize_t(o, PyExc_OverflowError);
    if (v == -1 && PyErr_Occurred()) return 0;
    switch (v) {
      case bob::sp::Extrapolation::BorderType::Zero:
      case bob::sp::Extrapolation::BorderType::Constant:
      case bob::sp::Extrapolation::BorderType::NearestNeighbour:
      case bob::sp::Extrapolation::BorderType::Circular:
      case bob::sp::Extrapolation::BorderType::Mirror:
        *b = static_cast<bob::sp::Extrapolation::BorderType>(v);
        return 1;
      default:
        PyErr_Format(PyExc_ValueError, "border parameter must be set to one of the integer values defined in `%s'", PyBobSpExtrapolationBorder_Type.tp_name);
        return 0;
    }
  } else {
    const std::string str = PyString_AsString(o);
    if (PyErr_Occurred()){
      PyErr_Format(PyExc_ValueError, "border parameter must be set to one of the integer values defined in `%s', or a string representation of it", PyBobSpExtrapolationBorder_Type.tp_name);
      return 0;
    }

    if (str == "Zero") *b = bob::sp::Extrapolation::BorderType::Zero;
    else if (str == "Constant") *b = bob::sp::Extrapolation::BorderType::Constant;
    else if (str == "NearestNeighbour") *b = bob::sp::Extrapolation::BorderType::NearestNeighbour;
    else if (str == "Circular") *b = bob::sp::Extrapolation::BorderType::Circular;
    else if (str == "Mirror") *b = bob::sp::Extrapolation::BorderType::Mirror;
    else {
      PyErr_Format(PyExc_ValueError, "border parameter must be set to one of the integer values defined in `%s', or a string representation of it", PyBobSpExtrapolationBorder_Type.tp_name);
      return 0;
    }
    return 1;
  }
}

static int PyBobSpExtrapolationBorder_Init(PyObject* self, PyObject*, PyObject*) {
  // Avoid instantiation of this class
  PyErr_Format(PyExc_NotImplementedError, "cannot initialize C++ enumeration bindings `%s' - use one of the class' attached attributes instead", Py_TYPE(self)->tp_name);
  return -1;
}


bool init_BobSpBorderType(PyObject* module){

  PyBobSpExtrapolationBorder_Type.tp_name = s_bordertype.name();
  PyBobSpExtrapolationBorder_Type.tp_basicsize = sizeof(PyBobSpExtrapolationBorder_Type);
  PyBobSpExtrapolationBorder_Type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
  PyBobSpExtrapolationBorder_Type.tp_doc = s_bordertype.doc();

  PyBobSpExtrapolationBorder_Type.tp_init = reinterpret_cast<initproc>(PyBobSpExtrapolationBorder_Init);
  PyBobSpExtrapolationBorder_Type.tp_dict = create_enumerations();

  // check that everything is fine
  if (PyType_Ready(&PyBobSpExtrapolationBorder_Type) < 0) return false;
  return PyModule_AddObject(module, "BorderType", Py_BuildValue("O", &PyBobSpExtrapolationBorder_Type)) >= 0;
}


bob::extension::FunctionDoc s_extrapolate = bob::extension::FunctionDoc(
  "extrapolate",
  "Extrapolates values in the given array using the specified border type",
  "This function extrapolates the given ``src`` array to the given ``dst`` array. "
  "The ``dst`` array needs to be at least as large as ``src``. "
  "First, the offset position is estimated and the ``src`` array is copied into the center of the ``dst`` array. "
  "Afterward, the remaining parts of the array is filled using the desired border handling, see :py:class:`BorderType` for details.\n\n"
  "This function works for 1D or 2D arrays. "
  "The parameter ``value`` is only used if the border type is set to :py:attr:`BorderType.Zero`. "
  "It is, by default, set to ``0.``, or the equivalent on the datatype passed as input. "
  "For example, ``False``, if the input is boolean and 0+0j, if it is complex."
)
.add_prototype("src, dst, [border], [value]")
.add_parameter("src", "array_like(1D or 2D)", "The input array that should be extrapolated")
.add_parameter("dst", "array_like(1D or 2D)", "The output array to write the result into; must be at least as large as ``src``")
.add_parameter("border", ":py:class:`BorderType` or str", "[Default: :py:attr:`BorderType.Zero`] The desired border handling")
.add_parameter("value", "same data type as in ``src``", "[Default: 0] The desired constant; only used for :py:attr:`BorderType.Constant`")
;
template <typename T> PyObject* inner_extrapolate (PyBlitzArrayObject* src,
    PyBlitzArrayObject* dst, bob::sp::Extrapolation::BorderType& border,
    PyObject* value) {

  //converts value into a proper scalar
  T c_value = static_cast<T>(0);
  if (value) {
    c_value = PyBlitzArrayCxx_AsCScalar<T>(value);
    if (PyErr_Occurred()) return 0;
  }

  switch (src->ndim) {
    case 1:
      bob::sp::extrapolate(*PyBlitzArrayCxx_AsBlitz<T,1>(src), *PyBlitzArrayCxx_AsBlitz<T,1>(dst), border, c_value);
      break;
    case 2:
      bob::sp::extrapolate(*PyBlitzArrayCxx_AsBlitz<T,2>(src), *PyBlitzArrayCxx_AsBlitz<T,2>(dst), border, c_value);
      break;
    default:
      PyErr_Format(PyExc_TypeError, "extrapolation does not support arrays with %" PY_FORMAT_SIZE_T "d dimensions", src->ndim);
      return 0;
  }
  Py_RETURN_NONE;
}

PyObject* PyBobSpExtrapolate(PyObject*, PyObject* args, PyObject* kwds) {
BOB_TRY
  char** kwlist = s_extrapolate.kwlist();

  PyBlitzArrayObject* src;
  PyBlitzArrayObject* dst;
  bob::sp::Extrapolation::BorderType border = bob::sp::Extrapolation::Zero;
  PyObject* value = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&O&|O&O",
        kwlist,
        &PyBlitzArray_Converter, &src,
        &PyBlitzArray_OutputConverter, &dst,
        &PyBobSpExtrapolationBorder_Converter, &border,
        &value)) return 0;

  auto src_ = make_safe(src);
  auto dst_ = make_safe(dst);

  if (src->type_num != dst->type_num) {
    PyErr_Format(PyExc_TypeError, "source and destination arrays must have the same data types (src: `%s' != dst: `%s')",
        PyBlitzArray_TypenumAsString(src->type_num),
        PyBlitzArray_TypenumAsString(dst->type_num));
    return 0;
  }

  if (src->ndim != dst->ndim) {
    PyErr_Format(PyExc_TypeError, "source and destination arrays must have the same number of dimensions types (src: `%" PY_FORMAT_SIZE_T "d' != dst: `%" PY_FORMAT_SIZE_T "d')", src->ndim, dst->ndim);
    return 0;
  }

  if (src->ndim != 1 && src->ndim != 2) {
    PyErr_Format(PyExc_TypeError, "source and destination arrays must have one or two dimensions, not `%" PY_FORMAT_SIZE_T "d", src->ndim);
    return 0;
  }

  switch (src->type_num) {
    case NPY_BOOL:
      return inner_extrapolate<bool>(src, dst, border, value);
    case NPY_INT8:
      return inner_extrapolate<int8_t>(src, dst, border, value);
    case NPY_INT16:
      return inner_extrapolate<int16_t>(src, dst, border, value);
    case NPY_INT32:
      return inner_extrapolate<int32_t>(src, dst, border, value);
    case NPY_INT64:
      return inner_extrapolate<int64_t>(src, dst, border, value);
    case NPY_UINT8:
      return inner_extrapolate<uint8_t>(src, dst, border, value);
    case NPY_UINT16:
      return inner_extrapolate<uint16_t>(src, dst, border, value);
    case NPY_UINT32:
      return inner_extrapolate<uint32_t>(src, dst, border, value);
    case NPY_UINT64:
      return inner_extrapolate<uint64_t>(src, dst, border, value);
    case NPY_FLOAT32:
      return inner_extrapolate<float>(src, dst, border, value);
    case NPY_FLOAT64:
      return inner_extrapolate<double>(src, dst, border, value);
    case NPY_COMPLEX64:
      return inner_extrapolate<std::complex<float>>(src, dst, border, value);
    case NPY_COMPLEX128:
      return inner_extrapolate<std::complex<double>>(src, dst, border, value);
    default:
      PyErr_Format(PyExc_TypeError, "extrapolation from `%s' (%d) is not supported", PyBlitzArray_TypenumAsString(src->type_num), src->type_num);
  }
  return 0;
BOB_CATCH_FUNCTION("extrapolate", 0)
}
