/**
 * @author Ivana Chingovska <ivana.chingovska@idiap.ch>
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Fri 31 Jan 14:26:40 2014
 *
 * @brief Support for quantization at our signal processing toolbox.
 *
 * @todo Use enumerations (see example in "extrapolate.cpp") instead of strings
 * as return value for `quantization_type'.
 *
 * @todo Clean-up: initialization code is pretty tricky. There are different
 * ways to initialize the functor which are pretty much disjoint. Maybe these
 * should be different types?
 *
 * @todo Extend: quantization class does not support generic input array.
 * Limited to uint16 and uint8. Output is always in uint32. Ideally, the output
 * should be dependent on the range the user wants to use. Input should be
 * arbitrary.
 */

#include "main.h"

static auto s_quantization = bob::extension::ClassDoc(
  BOB_EXT_MODULE_PREFIX ".Quantization",
  "Functor to quantize 1D or 2D signals into different number of levels",
  "At the moment, only ``uint8`` and ``uint16`` data types are supported. "
  "The output array returned by this functor will always have a ``uint32`` data type."
).add_constructor(bob::extension::FunctionDoc(
  "Quantization",
  "Creates a new Quantization functor for the given data type",
  "The functor can be created in two different ways. "
  "In the first way, quantization is performed by quantizing into the full range of levels for the given data type. "
  "The total number of levels can be limited using the ``num_levels``, ``min_level`` and ``max_level`` parameters, see their documentation.\n\n"
  "The second constructor takes the quantization table, which can be used for non-uniform quantization. "
  "Each element of the ``quantization_table`` corresponds to the lower boundary of the particular quantization level. "
  "E.g. ``numpy.array([ 0,  5, 10])`` means quantization in 3 levels. "
  "Input values in the range :math:`[0,4]` will be quantized to level 0, input values in the range :math:`[5,9]` will be  quantized to level 1 and input values in the range :math:`[10-\\text{max}]` will be quantized to level 2.\n\n"
  "The third constructor, finally, copies the quantization table from the given ``other`` :py:class:`Quantization` object."
)
.add_prototype("dtype, [rounding], [num_levels], [min_level], [max_level]", "")
.add_prototype("quantization_table", "")
.add_prototype("other", "")
.add_parameter("dtype", ":py:class:`numpy.dtype` or anything convertible", "The data type of arrays that are going to be **input** by this functor; Currently supported are ``uint8`` and ``uint16``")
.add_parameter("rounding", "bool", "[Default: ``False``] If set to ``True`` (defaults to ``False``), performs Matlab-like uniform quantization with rounding (see http://www.mathworks.com/matlabcentral/newsreader/view_thread/275291)")
.add_parameter("num_levels", "int", "[Default: -1] The number of quantization levels. The default is the total number of discrete values permitted by the ``dtype``")
.add_parameter("min_level", "dtype", "Input values smaller than or equal to this value are scaled to this value prior to quantization --> they will be scaled in the lowest quantization level")
.add_parameter("max_level", "dtype", "Input values higher than this value are scaled to this value prior to quantization -->they will be scaled in the highest quantization level")
.add_parameter("quantization_table", "array_like(1D, dtype)", "A 1-dimensional containing user-specified thresholds for the quantization")
.add_parameter("other", ":py:class:`Quantization`", "another Quantization object to create a deep-copy of")
);

PyTypeObject PyBobSpQuantization_Type{
    PyVarObject_HEAD_INIT(0, 0)
    0
};

int PyBobSpQuantization_Check(PyObject* o) {
  return PyObject_IsInstance(o, reinterpret_cast<PyObject*>(&PyBobSpQuantization_Type));
}

static void PyBobSpQuantization_Delete (PyBobSpQuantizationObject* self) {
  self->cxx.reset();
  Py_TYPE(self)->tp_free((PyObject*)self);
}


static int PyBobSpQuantization_InitCopy(PyBobSpQuantizationObject* self, PyObject* args, PyObject* kwds) {
  char** kwlist = s_quantization.kwlist(2);

  PyBobSpQuantizationObject* other;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!", kwlist,
        &PyBobSpQuantization_Type, &other)) return -1;

  self->type_num = other->type_num;
  switch (self->type_num) {
    case NPY_UINT8:
      self->cxx.reset(new bob::sp::Quantization<uint8_t>(*boost::static_pointer_cast<bob::sp::Quantization<uint8_t>>(other->cxx)));
    case NPY_UINT16:
      self->cxx.reset(new bob::sp::Quantization<uint16_t>(*boost::static_pointer_cast<bob::sp::Quantization<uint16_t>>(other->cxx)));
    default:
      PyErr_Format(PyExc_TypeError, "`%s' only accepts `uint8' or `uint16' as data types (not `%s')", Py_TYPE(self)->tp_name, PyBlitzArray_TypenumAsString(other->type_num));
      return -1;
  }
  return 0; ///< SUCCESS
}

template <typename T>
int initialize(PyBobSpQuantizationObject* self, bob::sp::quantization::QuantizationType type, Py_ssize_t levels, PyObject* min, PyObject* max) {
  // calculates all missing elements:
  T c_min = std::numeric_limits<T>::min();
  if (min) {
    c_min = PyBlitzArrayCxx_AsCScalar<T>(min);
    if (PyErr_Occurred()) return -1;
  }

  T c_max = std::numeric_limits<T>::max();
  if (max) {
    c_max = PyBlitzArrayCxx_AsCScalar<T>(max);
    if (PyErr_Occurred()) return -1;
  }

  if (levels <= 0) levels = c_max - c_min + 1;

  self->cxx.reset(new bob::sp::Quantization<T>(type, levels, c_min, c_max));
  return 0; ///< SUCCESS
}

static int PyBobSpQuantization_InitDiscrete(PyBobSpQuantizationObject* self, PyObject *args, PyObject* kwds) {
  char** kwlist = s_quantization.kwlist(0);

  int type_num;
  PyObject* rounding = Py_False;
  Py_ssize_t levels = -1;
  PyObject* min = 0;
  PyObject* max = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&|O!nOO", kwlist,
        &PyBlitzArray_TypenumConverter, &type_num,
        &PyBool_Type, &rounding,
        &levels,
        &min,
        &max
        )) return -1;

  bob::sp::quantization::QuantizationType rounding_enum = PyObject_IsTrue(rounding) ? bob::sp::quantization::UNIFORM_ROUNDING : bob::sp::quantization::UNIFORM;

  self->type_num = type_num;
  switch (type_num) {
    case NPY_UINT8:
      return initialize<uint8_t>(self, rounding_enum, levels, min, max);
    case NPY_UINT16:
      return initialize<uint16_t>(self, rounding_enum, levels, min, max);
    default:
      PyErr_Format(PyExc_TypeError, "`%s' only accepts `uint8' or `uint16' as data types (not `%s')", Py_TYPE(self)->tp_name, PyBlitzArray_TypenumAsString(type_num));
  }

  return -1; ///< FAIULRE
}

static int PyBobSpQuantization_InitTable(PyBobSpQuantizationObject* self, PyObject *args, PyObject* kwds) {
  char** kwlist = s_quantization.kwlist(1);

  PyBlitzArrayObject* table;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&", kwlist,
        &PyBlitzArray_Converter, &table)) return -1;

  auto table_ = make_safe(table);

  if (table->ndim != 1) {
    PyErr_Format(PyExc_TypeError, "`%s' only accepts 1-dimensional arrays as quantization table (not %" PY_FORMAT_SIZE_T "dD arrays)", Py_TYPE(self)->tp_name, table->ndim);
    return -1;
  }

  switch (table->type_num) {
    case NPY_UINT8:
      self->cxx.reset(new bob::sp::Quantization<uint8_t>(*PyBlitzArrayCxx_AsBlitz<uint8_t,1>(table)));
      break;
    case NPY_UINT16:
      self->cxx.reset(new bob::sp::Quantization<uint16_t>(*PyBlitzArrayCxx_AsBlitz<uint16_t,1>(table)));
      break;
    default:
      PyErr_Format(PyExc_TypeError, "`%s' only accepts 1-dimensional `uint8' or `uint16' arrays as quantization tables (not `%s' arrays)", Py_TYPE(self)->tp_name, PyBlitzArray_TypenumAsString(table->type_num));
      return -1;
  }
  self->type_num = table->type_num;
  return 0; ///< SUCCESS
}

static int PyBobSpQuantization_Init(PyBobSpQuantizationObject* self, PyObject* args, PyObject* kwds) {
BOB_TRY
  Py_ssize_t nargs = (args?PyTuple_Size(args):0) + (kwds?PyDict_Size(kwds):0);

  switch (nargs) {
    case 1:
      {
        PyObject* arg = 0; ///< borrowed (don't delete)
        if (PyTuple_Size(args)) arg = PyTuple_GET_ITEM(args, 0);
        else {
          PyObject* tmp = PyDict_Values(kwds);
          auto tmp_ = make_safe(tmp);
          arg = PyList_GET_ITEM(tmp, 0);
        }

        if (PyBlitzArray_Check(arg) || PyArray_Check(arg))
          return PyBobSpQuantization_InitTable(self, args, kwds);
        else if (PyBobSpQuantization_Check(arg))
          return PyBobSpQuantization_InitCopy(self, args, kwds);
        else
          return PyBobSpQuantization_InitDiscrete(self, args, kwds);

      }
      break;

    case 2:
    case 3:
    case 4:
    case 5:
      return PyBobSpQuantization_InitDiscrete(self, args, kwds);

    default:
      PyErr_Format(PyExc_RuntimeError, "number of arguments mismatch - %s requires 1, 2, 3, 4 or 5 arguments for initialization, but you provided %" PY_FORMAT_SIZE_T "d (see help)", Py_TYPE(self)->tp_name, nargs);
  }

  return -1;
BOB_CATCH_MEMBER("constructor", -1)
}

static auto s_dtype = bob::extension::VariableDoc(
  "dtype",
  ":py:class:`numpy.dtype`",
  "The data type of arrays that are going to be **input** by this functor",
  "Currently supported values are ``uint8`` and ``uint16``"
);

static PyObject* PyBobSpQuantization_GetDtype(PyBobSpQuantizationObject* self, void* /*closure*/) {
BOB_TRY
  PyArray_Descr* retval = PyArray_DescrFromType(self->type_num);
  return Py_BuildValue("O", retval);
BOB_CATCH_MEMBER("dtype", 0)
}


static auto s_quantization_type = bob::extension::VariableDoc(
  "quantization_type",
  "str",
  "The type of quantization that is currently performed",
  "Possible values of this parameter are:\n\n"
  "* ``uniform``: uniform quantization of the input signal within the range between :py:attr:`min_level` and :py:attr:`max_level`\n\n"
  "* ``uniform_rounding``: same as ``uniform``, but implemented in a similar way to Matlab quantization (see http://www.mathworks.com/matlabcentral/newsreader/view_thread/275291)\n\n"
  "* ``user_spec``: quantization according to user-specified :py:attr`quantization_table` of thresholds"
);
static PyObject* PyBobSpQuantization_GetQuantizationType(PyBobSpQuantizationObject* self, void* /*closure*/) {
BOB_TRY
  bob::sp::quantization::QuantizationType type;

  switch(self->type_num) {
    case NPY_UINT8:
      type = boost::static_pointer_cast<bob::sp::Quantization<uint8_t>>(self->cxx)->getType();
      break;
    case NPY_UINT16:
      type = boost::static_pointer_cast<bob::sp::Quantization<uint8_t>>(self->cxx)->getType();
      break;
    default:
      PyErr_Format(PyExc_RuntimeError, "don't know how to cope with `%s' object with dtype == `%s' -- DEBUG ME", Py_TYPE(self)->tp_name, PyBlitzArray_TypenumAsString(self->type_num));
      return 0;
  }

  switch(type) {
    case bob::sp::quantization::UNIFORM:
      return Py_BuildValue("s", "uniform");
    case bob::sp::quantization::UNIFORM_ROUNDING:
      return Py_BuildValue("s", "uniform_rounding");
    case bob::sp::quantization::USER_SPEC:
      return Py_BuildValue("s", "user_spec");
    default:
      PyErr_Format(PyExc_RuntimeError, "don't know how to cope with `%s' object with quantization method == `%d' -- DEBUG ME", Py_TYPE(self)->tp_name, (int)type);
  }

  return 0;
BOB_CATCH_MEMBER("quantization_type", 0)
}

static auto s_num_levels = bob::extension::VariableDoc(
  "num_levels",
  "int",
  "The number of quantization levels"
);
static PyObject* PyBobSpQuantization_GetNumLevels(PyBobSpQuantizationObject* self, void* /*closure*/) {
BOB_TRY
  Py_ssize_t v;

  switch(self->type_num) {
    case NPY_UINT8:
      v = boost::static_pointer_cast<bob::sp::Quantization<uint8_t>>(self->cxx)->getNumLevels();
      break;
    case NPY_UINT16:
      v = boost::static_pointer_cast<bob::sp::Quantization<uint8_t>>(self->cxx)->getNumLevels();
      break;
    default:
      PyErr_Format(PyExc_RuntimeError, "don't know how to cope with `%s' object with dtype == `%s' -- DEBUG ME", Py_TYPE(self)->tp_name, PyBlitzArray_TypenumAsString(self->type_num));
      return 0;
  }

  return Py_BuildValue("n", v);
BOB_CATCH_MEMBER("num_levels", 0)
}


static auto s_min_level = bob::extension::VariableDoc(
  "min_level",
  "dtype",
  "Input values smaller than or equal to this value are scaled to this value prior to quantization",
  "As a result, they will be scaled in the lowest quantization level. "
  "The data type of this scalar should be coercible to the datatype of the input."
);
static PyObject* PyBobSpQuantization_GetMinLevel(PyBobSpQuantizationObject* self, void* /*closure*/) {
BOB_TRY
  switch(self->type_num) {
    case NPY_UINT8:
      return PyBlitzArrayCxx_FromCScalar<uint8_t>(boost::static_pointer_cast<bob::sp::Quantization<uint8_t>>(self->cxx)->getMinLevel());
      break;
    case NPY_UINT16:
      return PyBlitzArrayCxx_FromCScalar<uint16_t>(boost::static_pointer_cast<bob::sp::Quantization<uint16_t>>(self->cxx)->getMinLevel());
      break;
    default:
      PyErr_Format(PyExc_RuntimeError, "don't know how to cope with `%s' object with dtype == `%s' -- DEBUG ME", Py_TYPE(self)->tp_name, PyBlitzArray_TypenumAsString(self->type_num));
  }

  return 0;
BOB_CATCH_MEMBER("min_level", 0)
}


static auto s_max_level = bob::extension::VariableDoc(
  "max_level",
  "dtype",
  "Input values higher than this value are scaled to this value prior to quantization",
  "As a result, they will be scaled in the highest quantization level. "
  "The data type of this scalar should be coercible to the datatype of the input"
);
static PyObject* PyBobSpQuantization_GetMaxLevel(PyBobSpQuantizationObject* self, void* /*closure*/) {
BOB_TRY
  switch(self->type_num) {
    case NPY_UINT8:
      return PyBlitzArrayCxx_FromCScalar<uint8_t>(boost::static_pointer_cast<bob::sp::Quantization<uint8_t>>(self->cxx)->getMaxLevel());
    case NPY_UINT16:
      return PyBlitzArrayCxx_FromCScalar<uint16_t>(boost::static_pointer_cast<bob::sp::Quantization<uint16_t>>(self->cxx)->getMaxLevel());
    default:
      PyErr_Format(PyExc_RuntimeError, "don't know how to cope with `%s' object with dtype == `%s' -- DEBUG ME", Py_TYPE(self)->tp_name, PyBlitzArray_TypenumAsString(self->type_num));
  }

  return 0;
BOB_CATCH_MEMBER("max_level", 0)
}


static auto s_quantization_table = bob::extension::VariableDoc(
  "quantization_table",
  "array_like(1D, dtype)",
  "A 1-dimensional array containing thresholds for the quantization",
  "Each element corresponds to the lower boundary of the particular quantization level. "
  "E.g. ``array([ 0,  5, 10])`` means quantization is performed in 3 levels. "
  "Input values in the range :math:`[0,4]` will be quantized to level 0, input values in the range :math:`[5,9]` will be quantized to level 1 and input values in the range :math:`[10-\\text{max}]` will be quantized to level 2."
);
static PyObject* PyBobSpQuantization_GetQuantizationTable(PyBobSpQuantizationObject* self, void* /*closure*/) {
BOB_TRY
  switch(self->type_num) {
    case NPY_UINT8:
      return PyBlitzArrayCxx_AsConstNumpy(boost::static_pointer_cast<bob::sp::Quantization<uint8_t>>(self->cxx)->getThresholds());
    case NPY_UINT16:
      return PyBlitzArrayCxx_AsConstNumpy(boost::static_pointer_cast<bob::sp::Quantization<uint16_t>>(self->cxx)->getThresholds());
    default:
      PyErr_Format(PyExc_RuntimeError, "don't know how to cope with `%s' object with dtype == `%s' -- DEBUG ME", Py_TYPE(self)->tp_name, PyBlitzArray_TypenumAsString(self->type_num));
  }
  return 0;
BOB_CATCH_MEMBER("quantization_table", 0)
}


static PyGetSetDef PyBobSpQuantization_getseters[] = {
    {
      s_dtype.name(),
      (getter)PyBobSpQuantization_GetDtype,
      0,
      s_dtype.doc(),
      0
    },
    {
      s_quantization_type.name(),
      (getter)PyBobSpQuantization_GetQuantizationType,
      0,
      s_quantization_type.doc(),
      0
    },
    {
      s_num_levels.name(),
      (getter)PyBobSpQuantization_GetNumLevels,
      0,
      s_num_levels.doc(),
      0
    },
    {
      s_min_level.name(),
      (getter)PyBobSpQuantization_GetMinLevel,
      0,
      s_min_level.doc(),
      0
    },
    {
      s_max_level.name(),
      (getter)PyBobSpQuantization_GetMaxLevel,
      0,
      s_max_level.doc(),
      0
    },
    {
      s_quantization_table.name(),
      (getter)PyBobSpQuantization_GetQuantizationTable,
      0,
      s_quantization_table.doc(),
      0
    },
    {0}  /* Sentinel */
};


static auto s_quantize = bob::extension::FunctionDoc(
  "quantize",
  "Quantizes the given input",
  ".. todo:: Document exaxtly, what this function does, i.e., what is stored in the output array.\n\n"
  ".. note:: The :py:meth:`__call__`` function is a synonym for this function, turning this object into a functor.",
  true
)
.add_prototype("input, [output]", "output")
.add_parameter("input", "array_like (1D or 2D, uint8 or uint16)", "The array to be quantized")
.add_parameter("output", "array_like(1D or 2D, uint32)", "The pre-allocated array where to store the output, with the same dimensions as ``input``")
.add_return("output", "array_like(1D or 2D, uint32)", "The resulting quantized data")
;
template <typename T>
static void call(PyBobSpQuantizationObject* self,  PyBlitzArrayObject* input, PyBlitzArrayObject* output) {

  auto op = boost::static_pointer_cast<bob::sp::Quantization<T>>(self->cxx);

  switch(input->ndim) {
    case 1:
      op->operator()(*PyBlitzArrayCxx_AsBlitz<T,1>(input),
          *PyBlitzArrayCxx_AsBlitz<uint32_t,1>(output));
      break;
    case 2:
      op->operator()(*PyBlitzArrayCxx_AsBlitz<T,2>(input),
          *PyBlitzArrayCxx_AsBlitz<uint32_t,2>(output));
      break;
    default:
      throw std::runtime_error("don't know how to cope with Quantization object with unknown dtype -- DEBUG ME");
  }
}

static PyObject* PyBobSpQuantization_Quantize(PyBobSpQuantizationObject* self, PyObject* args, PyObject* kwds) {
BOB_TRY
  char** kwlist = s_quantize.kwlist();

  PyBlitzArrayObject* input;
  PyBlitzArrayObject* output = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&|O&", kwlist,
        &PyBlitzArray_Converter, &input,
        &PyBlitzArray_OutputConverter, &output
        )) return 0;

  //protects acquired resources through this scope
  auto input_ = make_safe(input);
  auto output_ = make_xsafe(output);

  if (self->type_num != input->type_num) {
    PyErr_Format(PyExc_TypeError, "`%s' only supports `%s' arrays for `input', not `%s'", Py_TYPE(self)->tp_name, PyBlitzArray_TypenumAsString(self->type_num),
        PyBlitzArray_TypenumAsString(input->type_num));
    return 0;
  }

  if (output && output->type_num != NPY_UINT32) {
    PyErr_Format(PyExc_TypeError, "`%s' only supports `uint32' arrays for `output', not `%s'", Py_TYPE(self)->tp_name, PyBlitzArray_TypenumAsString(output->type_num));
    return 0;
  }

  if (input->ndim < 1 || input->ndim > 2) {
    PyErr_Format(PyExc_TypeError, "`%s' only accepts 1 or 2-dimensional arrays (not %" PY_FORMAT_SIZE_T "dD arrays)", Py_TYPE(self)->tp_name, input->ndim);
    return 0;
  }

  if (output && input->ndim != output->ndim) {
    PyErr_Format(PyExc_RuntimeError, "Input and output arrays should have matching number of dimensions, but input array `input' has %" PY_FORMAT_SIZE_T "d dimensions while output array `output' has %" PY_FORMAT_SIZE_T "d dimensions", input->ndim, output->ndim);
    return 0;
  }

  if (input->ndim == 1) {
    if (output && output->shape[0] != input->shape[0]) {
      PyErr_Format(PyExc_RuntimeError, "1D `output' array should have %" PY_FORMAT_SIZE_T "d elements matching `%s' input size, not %" PY_FORMAT_SIZE_T "d elements", input->shape[0], Py_TYPE(self)->tp_name, output->shape[0]);
      return 0;
    }
  } else {
    if (output && output->shape[1] != input->shape[1]) {
      PyErr_Format(PyExc_RuntimeError, "2D `output' array should have %" PY_FORMAT_SIZE_T "d columns matching input size, not %" PY_FORMAT_SIZE_T "d columns", input->shape[1], output->shape[1]);
      return 0;
    }
    if (output && input->shape[0] != output->shape[0]) {
      PyErr_Format(PyExc_RuntimeError, "2D `output' array should have %" PY_FORMAT_SIZE_T "d rows matching `input' size, not %" PY_FORMAT_SIZE_T "d rows", input->shape[0], output->shape[0]);
      return 0;
    }
  }

  /** if ``output`` was not pre-allocated, do it now **/
  if (!output) {
    output = (PyBlitzArrayObject*)PyBlitzArray_SimpleNew(NPY_UINT32, input->ndim, input->shape);
    output_ = make_safe(output);
  }

  /** all basic checks are done, can call the functor now **/
  switch (self->type_num) {
    case NPY_UINT8:
      call<uint8_t>(self, input, output);
      break;
    case NPY_UINT16:
      call<uint16_t>(self, input, output);
      break;
    default:
      PyErr_Format(PyExc_RuntimeError, "don't know how to cope with `%s' object with dtype == `%s' -- DEBUG ME", Py_TYPE(self)->tp_name, PyBlitzArray_TypenumAsString(self->type_num));
      return 0;
  }

  return PyBlitzArray_AsNumpyArray(output, 0);
BOB_CATCH_MEMBER("quantize", 0)
}


static auto s_quantization_level = bob::extension::FunctionDoc(
  "quantization_level",
  "Calculates the quantization level for a single input value",
  0,
  true
)
.add_prototype("input", "quantized")
.add_parameter("input", "dtype", "The value to be quantized")
.add_return("quantized", "dtype", "The resulting quantized value")
;
static PyObject* PyBobSpQuantization_QuantizationLevel(PyBobSpQuantizationObject* self, PyObject* args, PyObject* kwds) {
BOB_TRY
  char** kwlist = s_quantization_level.kwlist();

  PyObject* input = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O", kwlist, &input)) return 0;

  switch (self->type_num) {
    case NPY_UINT8:
      {
        auto c_input = PyBlitzArrayCxx_AsCScalar<uint8_t>(input);
        if (PyErr_Occurred()) return 0;
        return PyBlitzArrayCxx_FromCScalar<uint8_t>(boost::static_pointer_cast<bob::sp::Quantization<uint8_t>>(self->cxx)->quantization_level(c_input));
      }
    case NPY_UINT16:
      {
        auto c_input = PyBlitzArrayCxx_AsCScalar<uint16_t>(input);
        if (PyErr_Occurred()) return 0;
        return PyBlitzArrayCxx_FromCScalar<uint8_t>(boost::static_pointer_cast<bob::sp::Quantization<uint16_t>>(self->cxx)->quantization_level(c_input));
      }
    default:
      PyErr_Format(PyExc_RuntimeError, "don't know how to cope with `%s' object with dtype == `%s' -- DEBUG ME", Py_TYPE(self)->tp_name, PyBlitzArray_TypenumAsString(self->type_num));
  }
  return 0;
BOB_CATCH_MEMBER("quantization_level", 0)
}


static PyMethodDef PyBobSpQuantization_methods[] = {
  {
    s_quantize.name(),
    (PyCFunction)PyBobSpQuantization_Quantize,
    METH_VARARGS|METH_KEYWORDS,
    s_quantize.doc(),
  },
  {
    s_quantization_level.name(),
    (PyCFunction)PyBobSpQuantization_QuantizationLevel,
    METH_VARARGS|METH_KEYWORDS,
    s_quantization_level.doc(),
  },
  {0} /* Sentinel */
};

bool init_BobSpQuantization(PyObject* module){
  // class definition
  PyBobSpQuantization_Type.tp_name = s_quantization.name();
  PyBobSpQuantization_Type.tp_basicsize = sizeof(PyBobSpQuantizationObject);
  PyBobSpQuantization_Type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
  PyBobSpQuantization_Type.tp_doc = s_quantization.doc();

  PyBobSpQuantization_Type.tp_new = PyType_GenericNew;
  PyBobSpQuantization_Type.tp_dealloc = reinterpret_cast<destructor>(PyBobSpQuantization_Delete);
  PyBobSpQuantization_Type.tp_init = reinterpret_cast<initproc>(PyBobSpQuantization_Init);
  PyBobSpQuantization_Type.tp_methods = PyBobSpQuantization_methods;
  PyBobSpQuantization_Type.tp_getset = PyBobSpQuantization_getseters;
  PyBobSpQuantization_Type.tp_call = reinterpret_cast<ternaryfunc>(PyBobSpQuantization_Quantize);

  // check that everything is fine
  if (PyType_Ready(&PyBobSpQuantization_Type) < 0) return false;
  return PyModule_AddObject(module, "Quantization", Py_BuildValue("O", &PyBobSpQuantization_Type)) >= 0;
}
