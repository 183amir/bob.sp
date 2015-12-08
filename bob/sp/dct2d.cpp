/**
 * @date Mon Jun 20 11:47:58 2011 +0200
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief Python bindings to statistical methods
 *
 * Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
 */

#include "main.h"

static auto s_dct2d = bob::extension::ClassDoc(
  BOB_EXT_MODULE_PREFIX ".DCT2D",
  "Calculates the direct DCT of a 2D array/signal",
  "Input and output arrays are 2D NumPy arrays of type ``float64``."
)
.add_constructor(bob::extension::FunctionDoc(
  "DCT2D",
  "Creates a new DCT2D extractor in the given dimensions"
)
.add_prototype("height, width", "")
.add_prototype("other", "")
.add_parameter("height, width", "int", "The size of the input signal this class will be able to handle")
.add_parameter("other", ":py:class:`DCT2D`", "The other DCT2D class to copy-construct")
);

PyTypeObject PyBobSpDCT2D_Type = {
  PyVarObject_HEAD_INIT(0, 0)
  0
};

int PyBobSpDCT2D_Check(PyObject* o) {
  return PyObject_IsInstance(o, reinterpret_cast<PyObject*>(&PyBobSpDCT2D_Type));
}

static void PyBobSpDCT2D_Delete (PyBobSpDCT2DObject* self) {
  self->cxx.reset();
  Py_TYPE(self)->tp_free((PyObject*)self);
}

static int PyBobSpDCT2D_InitCopy(PyBobSpDCT2DObject* self, PyObject* args, PyObject* kwds) {
  char** kwlist = s_dct2d.kwlist(1);
  PyBobSpDCT2DObject* other;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!", kwlist, &PyBobSpDCT2D_Type, &other)) return -1;

  self->cxx.reset(new bob::sp::DCT2D(*(other->cxx)));
  return 0;
}

static int PyBobSpDCT2D_InitShape(PyBobSpDCT2DObject* self, PyObject *args, PyObject* kwds) {
  char** kwlist = s_dct2d.kwlist(0);

  Py_ssize_t h;
  Py_ssize_t w;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "nn", kwlist, &h, &w)) return -1;

  self->cxx.reset(new bob::sp::DCT2D(h, w));
  return 0; ///< SUCCESS
}

static int PyBobSpDCT2D_Init(PyBobSpDCT2DObject* self, PyObject* args, PyObject* kwds) {
BOB_TRY
  Py_ssize_t nargs = (args?PyTuple_Size(args):0) + (kwds?PyDict_Size(kwds):0);

  switch (nargs) {
    case 1:
      return PyBobSpDCT2D_InitCopy(self, args, kwds);

    case 2:
      return PyBobSpDCT2D_InitShape(self, args, kwds);

    default:
      PyErr_Format(PyExc_RuntimeError, "number of arguments mismatch - %s requires 1 argument, but you provided %" PY_FORMAT_SIZE_T "d (see help)", Py_TYPE(self)->tp_name, nargs);
  }
  return -1;
BOB_CATCH_MEMBER("constructor", -1)
}

static PyObject* PyBobSpDCT2D_Repr(PyBobSpDCT2DObject* self) {
BOB_TRY
  return PyString_FromFormat("%s(height=%zu, width=%zu)", Py_TYPE(self)->tp_name, self->cxx->getHeight(), self->cxx->getWidth());
BOB_CATCH_MEMBER("__repr__", 0)
}

static PyObject* PyBobSpDCT2D_RichCompare (PyBobSpDCT2DObject* self, PyObject* other, int op) {
BOB_TRY
  if (!PyBobSpDCT2D_Check(other)) {
    PyErr_Format(PyExc_TypeError, "cannot compare `%s' with `%s'",
        Py_TYPE(self)->tp_name, Py_TYPE(other)->tp_name);
    return 0;
  }

  auto other_ = reinterpret_cast<PyBobSpDCT2DObject*>(other);

  switch (op) {
    case Py_EQ:
      if (self->cxx->operator==(*other_->cxx)) Py_RETURN_TRUE;
      Py_RETURN_FALSE;
      break;
    case Py_NE:
      if (self->cxx->operator!=(*other_->cxx)) Py_RETURN_TRUE;
      Py_RETURN_FALSE;
      break;
    default:
      Py_INCREF(Py_NotImplemented);
      return Py_NotImplemented;
  }
BOB_CATCH_MEMBER("RichCompare", 0)
}

static auto s_height = bob::extension::VariableDoc(
  "height",
  "int",
  "The height of the input and output data"
);
static PyObject* PyBobSpDCT2D_GetHeight(PyBobSpDCT2DObject* self, void* /*closure*/) {
BOB_TRY
  return Py_BuildValue("n", self->cxx->getHeight());
BOB_CATCH_MEMBER("height", 0)
}

static int PyBobSpDCT2D_SetHeight(PyBobSpDCT2DObject* self, PyObject* o, void* /*closure*/) {
BOB_TRY
  if (!PyBob_NumberCheck(o)) {
    PyErr_Format(PyExc_TypeError, "`%s' height can only be set using a number, not `%s'", Py_TYPE(self)->tp_name, Py_TYPE(o)->tp_name);
    return -1;
  }

  Py_ssize_t len = PyNumber_AsSsize_t(o, PyExc_OverflowError);
  if (PyErr_Occurred()) return -1;

  self->cxx->setHeight(len);
  return 0;
BOB_CATCH_MEMBER("height", -1)
}

static auto s_width = bob::extension::VariableDoc(
  "width",
  "int",
  "The width of the input and output data"
);
static PyObject* PyBobSpDCT2D_GetWidth
(PyBobSpDCT2DObject* self, void* /*closure*/) {
BOB_TRY
  return Py_BuildValue("n", self->cxx->getWidth());
BOB_CATCH_MEMBER("width", 0)
}

static int PyBobSpDCT2D_SetWidth
(PyBobSpDCT2DObject* self, PyObject* o, void* /*closure*/) {
BOB_TRY

  if (!PyBob_NumberCheck(o)) {
    PyErr_Format(PyExc_TypeError, "`%s' width can only be set using a number, not `%s'", Py_TYPE(self)->tp_name, Py_TYPE(o)->tp_name);
    return -1;
  }

  Py_ssize_t len = PyNumber_AsSsize_t(o, PyExc_OverflowError);
  if (PyErr_Occurred()) return -1;

  self->cxx->setWidth(len);
  return 0;
BOB_CATCH_MEMBER("width", -1)
}

static auto s_shape = bob::extension::VariableDoc(
  "shape",
  "(int, int)",
  "A 2D tuple that represents the size of the input/output data"
);
static PyObject* PyBobSpDCT2D_GetShape(PyBobSpDCT2DObject* self, void* /*closure*/) {
BOB_TRY
  return Py_BuildValue("(nn)", self->cxx->getHeight(), self->cxx->getWidth());
BOB_CATCH_MEMBER("shape", 0)
}

static int PyBobSpDCT2D_SetShape(PyBobSpDCT2DObject* self, PyObject* o, void* /*closure*/) {
BOB_TRY
  if (!PySequence_Check(o)) {
    PyErr_Format(PyExc_TypeError, "`%s' shape can only be set using tuples (or sequences), not `%s'", Py_TYPE(self)->tp_name, Py_TYPE(o)->tp_name);
    return -1;
  }

  PyObject* shape = PySequence_Tuple(o);
  auto shape_ = make_safe(shape);

  if (PyTuple_GET_SIZE(shape) != 2) {
    PyErr_Format(PyExc_RuntimeError, "`%s' shape can only be set using 2-position tuples (or sequences), not an %" PY_FORMAT_SIZE_T "d-position sequence", Py_TYPE(self)->tp_name, PyTuple_GET_SIZE(shape));
    return -1;
  }

  Py_ssize_t h = PyNumber_AsSsize_t(PyTuple_GET_ITEM(shape, 0), PyExc_OverflowError);
  if (PyErr_Occurred()) return -1;
  Py_ssize_t w = PyNumber_AsSsize_t(PyTuple_GET_ITEM(shape, 1), PyExc_OverflowError);
  if (PyErr_Occurred()) return -1;

  self->cxx->setHeight(h);
  self->cxx->setWidth(w);
  return 0;
BOB_CATCH_MEMBER("shape", -1)
}

static PyGetSetDef PyBobSpDCT2D_getseters[] = {
    {
      s_height.name(),
      (getter)PyBobSpDCT2D_GetHeight,
      (setter)PyBobSpDCT2D_SetHeight,
      s_height.doc(),
      0
    },
    {
      s_width.name(),
      (getter)PyBobSpDCT2D_GetWidth,
      (setter)PyBobSpDCT2D_SetWidth,
      s_width.doc(),
      0
    },
    {
      s_shape.name(),
      (getter)PyBobSpDCT2D_GetShape,
      (setter)PyBobSpDCT2D_SetShape,
      s_shape.doc(),
      0
    },
    {0}  /* Sentinel */
};

static auto s_transform = bob::extension::FunctionDoc(
  "transform",
  "Calculates the direct DCT of the given signal",
  ".. note:: The :py:meth:`__call__` function is a synonym for this function.",
  true
)
.add_prototype("input, [output]", "output")
.add_parameter("input", "array_like(2D, float)", "The given input array. It must be of shape :py:attr:`shape`")
.add_parameter("output", "array_like(2D, float)", "A pre-allocated output array. If given, it must be of shape :py:attr:`shape`")
.add_return("output", "array_like(2D, float)", "The DCT result; identical to the ``output`` parameter, if given")
;
static PyObject* PyBobSpDCT2D_transform(PyBobSpDCT2DObject* self, PyObject* args, PyObject* kwds) {
BOB_TRY
  char** kwlist = s_transform.kwlist();

  PyBlitzArrayObject* input;
  PyBlitzArrayObject* output = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&|O&", kwlist,
        &PyBlitzArray_Converter, &input,
        &PyBlitzArray_OutputConverter, &output
        )) return 0;

  //protects acquired resources through this scope
  auto input_ = make_safe(input);
  auto output_ = make_xsafe(output);

  if (input->type_num != NPY_FLOAT64) {
    PyErr_Format(PyExc_TypeError, "`%s' only supports 64-bit float arrays for input array `input'", Py_TYPE(self)->tp_name);
    return 0;
  }

  if (output && output->type_num != NPY_FLOAT64) {
    PyErr_Format(PyExc_TypeError, "`%s' only supports 64-bit float arrays for output array `output'", Py_TYPE(self)->tp_name);
    return 0;
  }

  if (input->ndim != 2) {
    PyErr_Format(PyExc_TypeError, "`%s' only accepts 2-dimensional arrays (not %" PY_FORMAT_SIZE_T "dD arrays)", Py_TYPE(self)->tp_name, input->ndim);
    return 0;
  }

  if (output && input->ndim != output->ndim) {
    PyErr_Format(PyExc_RuntimeError, "Input and output arrays should have matching number of dimensions, but input array `input' has %" PY_FORMAT_SIZE_T "d dimensions while output array `output' has %" PY_FORMAT_SIZE_T "d dimensions", input->ndim, output->ndim);
    return 0;
  }

  if (output && output->shape[0] != (Py_ssize_t)self->cxx->getHeight()) {
    PyErr_Format(PyExc_RuntimeError, "2D `output' array should have %" PY_FORMAT_SIZE_T "d rows matching `%s' output size, not %" PY_FORMAT_SIZE_T "d elements", self->cxx->getHeight(), Py_TYPE(self)->tp_name, output->shape[0]);
    return 0;
  }

  if (output && output->shape[1] != (Py_ssize_t)self->cxx->getWidth()) {
    PyErr_Format(PyExc_RuntimeError, "2D `output' array should have %" PY_FORMAT_SIZE_T "d columns matching `%s' output size, not %" PY_FORMAT_SIZE_T "d elements", self->cxx->getWidth(), Py_TYPE(self)->tp_name, output->shape[1]);
    return 0;
  }

  /** if ``output`` was not pre-allocated, do it now **/
  if (!output) {
    Py_ssize_t size[2] = {static_cast<Py_ssize_t>(self->cxx->getHeight()), static_cast<Py_ssize_t>(self->cxx->getWidth())};
    output = (PyBlitzArrayObject*)PyBlitzArray_SimpleNew(NPY_FLOAT64, 2, size);
    output_ = make_safe(output);
  }

  /** all basic checks are done, can call the operator now **/
  self->cxx->operator()(*PyBlitzArrayCxx_AsBlitz<double,2>(input), *PyBlitzArrayCxx_AsBlitz<double,2>(output));

  return PyBlitzArray_AsNumpyArray(output, 0);
BOB_CATCH_MEMBER("transform", 0)
}

static PyMethodDef PyBobSpDCT2D_methods[] = {
    {
      s_transform.name(),
      (PyCFunction)PyBobSpDCT2D_transform,
      METH_VARARGS|METH_KEYWORDS,
      s_transform.doc()
    },
    {0}  /* Sentinel */
};

bool init_BobSpDCT2D(PyObject* module){
  // class definition
  PyBobSpDCT2D_Type.tp_name = s_dct2d.name();
  PyBobSpDCT2D_Type.tp_basicsize = sizeof(PyBobSpDCT2DObject);
  PyBobSpDCT2D_Type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
  PyBobSpDCT2D_Type.tp_doc = s_dct2d.doc();

  PyBobSpDCT2D_Type.tp_new = PyType_GenericNew;
  PyBobSpDCT2D_Type.tp_dealloc = reinterpret_cast<destructor>(PyBobSpDCT2D_Delete);
  PyBobSpDCT2D_Type.tp_init = reinterpret_cast<initproc>(PyBobSpDCT2D_Init);
  PyBobSpDCT2D_Type.tp_richcompare = reinterpret_cast<richcmpfunc>(PyBobSpDCT2D_RichCompare);
  PyBobSpDCT2D_Type.tp_methods = PyBobSpDCT2D_methods;
  PyBobSpDCT2D_Type.tp_getset = PyBobSpDCT2D_getseters;
  PyBobSpDCT2D_Type.tp_call = reinterpret_cast<ternaryfunc>(PyBobSpDCT2D_transform);
  PyBobSpDCT2D_Type.tp_repr = reinterpret_cast<reprfunc>(PyBobSpDCT2D_Repr);

  // check that everything is fine
  if (PyType_Ready(&PyBobSpDCT2D_Type) < 0) return false;
  return PyModule_AddObject(module, "DCT2D", Py_BuildValue("O", &PyBobSpDCT2D_Type)) >= 0;
}
