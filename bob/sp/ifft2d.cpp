/**
 * @date Mon Jun 20 11:47:58 2011 +0200
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief Python bindings to statistical methods
 *
 * Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
 */

#include "main.h"

static auto s_ifft2d = bob::extension::ClassDoc(
  BOB_EXT_MODULE_PREFIX ".IFFT2D",
  "Calculates the inverse FFT of a 2D array/signal",
  "Input and output arrays are 2D NumPy arrays of type ``complex128``."
)
.add_constructor(bob::extension::FunctionDoc(
  "IFFT2D",
  "Creates a new IFFT2D extractor in the given dimensions"
)
.add_prototype("height, width", "")
.add_prototype("other", "")
.add_parameter("height, width", "int", "The size of the input signal this class will be able to handle")
.add_parameter("other", ":py:class:`IFFT2D`", "The other IFFT2D class to copy-construct")
);

PyTypeObject PyBobSpIFFT2D_Type = {
  PyVarObject_HEAD_INIT(0, 0)
  0
};

int PyBobSpIFFT2D_Check(PyObject* o) {
  return PyObject_IsInstance(o, reinterpret_cast<PyObject*>(&PyBobSpIFFT2D_Type));
}

static void PyBobSpIFFT2D_Delete (PyBobSpIFFT2DObject* self) {
  self->cxx.reset();
  Py_TYPE(self)->tp_free((PyObject*)self);
}

static int PyBobSpIFFT2D_InitCopy(PyBobSpIFFT2DObject* self, PyObject* args, PyObject* kwds) {
  char** kwlist = s_ifft2d.kwlist(1);
  PyBobSpIFFT2DObject* other;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!", kwlist, &PyBobSpIFFT2D_Type, &other)) return -1;

  self->cxx.reset(new bob::sp::IFFT2D(*(other->cxx)));
  return 0;
}

static int PyBobSpIFFT2D_InitShape(PyBobSpIFFT2DObject* self, PyObject *args, PyObject* kwds) {
  char** kwlist = s_ifft2d.kwlist(0);

  Py_ssize_t h;
  Py_ssize_t w;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "nn", kwlist, &h, &w)) return -1;

  self->cxx.reset(new bob::sp::IFFT2D(h, w));
  return 0; ///< SUCCESS
}

static int PyBobSpIFFT2D_Init(PyBobSpIFFT2DObject* self, PyObject* args, PyObject* kwds) {
BOB_TRY
  Py_ssize_t nargs = (args?PyTuple_Size(args):0) + (kwds?PyDict_Size(kwds):0);

  switch (nargs) {
    case 1:
      return PyBobSpIFFT2D_InitCopy(self, args, kwds);

    case 2:
      return PyBobSpIFFT2D_InitShape(self, args, kwds);

    default:
      PyErr_Format(PyExc_RuntimeError, "number of arguments mismatch - %s requires 1 argument, but you provided %" PY_FORMAT_SIZE_T "d (see help)", Py_TYPE(self)->tp_name, nargs);
  }
  return -1;
BOB_CATCH_MEMBER("constructor", -1)
}

static PyObject* PyBobSpIFFT2D_Repr(PyBobSpIFFT2DObject* self) {
BOB_TRY
  return PyString_FromFormat("%s(height=%zu, width=%zu)", Py_TYPE(self)->tp_name, self->cxx->getHeight(), self->cxx->getWidth());
BOB_CATCH_MEMBER("__repr__", 0)
}

static PyObject* PyBobSpIFFT2D_RichCompare (PyBobSpIFFT2DObject* self, PyObject* other, int op) {
BOB_TRY
  if (!PyBobSpIFFT2D_Check(other)) {
    PyErr_Format(PyExc_TypeError, "cannot compare `%s' with `%s'",
        Py_TYPE(self)->tp_name, Py_TYPE(other)->tp_name);
    return 0;
  }

  auto other_ = reinterpret_cast<PyBobSpIFFT2DObject*>(other);

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
static PyObject* PyBobSpIFFT2D_GetHeight(PyBobSpIFFT2DObject* self, void* /*closure*/) {
BOB_TRY
  return Py_BuildValue("n", self->cxx->getHeight());
BOB_CATCH_MEMBER("height", 0)
}

static int PyBobSpIFFT2D_SetHeight(PyBobSpIFFT2DObject* self, PyObject* o, void* /*closure*/) {
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
static PyObject* PyBobSpIFFT2D_GetWidth(PyBobSpIFFT2DObject* self, void* /*closure*/) {
BOB_TRY
  return Py_BuildValue("n", self->cxx->getWidth());
BOB_CATCH_MEMBER("width", 0)
}

static int PyBobSpIFFT2D_SetWidth(PyBobSpIFFT2DObject* self, PyObject* o, void* /*closure*/) {
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
static PyObject* PyBobSpIFFT2D_GetShape(PyBobSpIFFT2DObject* self, void* /*closure*/) {
BOB_TRY
  return Py_BuildValue("(nn)", self->cxx->getHeight(), self->cxx->getWidth());
BOB_CATCH_MEMBER("shape", 0)
}

static int PyBobSpIFFT2D_SetShape(PyBobSpIFFT2DObject* self, PyObject* o, void* /*closure*/) {
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

static PyGetSetDef PyBobSpIFFT2D_getseters[] = {
    {
      s_height.name(),
      (getter)PyBobSpIFFT2D_GetHeight,
      (setter)PyBobSpIFFT2D_SetHeight,
      s_height.doc(),
      0
    },
    {
      s_width.name(),
      (getter)PyBobSpIFFT2D_GetWidth,
      (setter)PyBobSpIFFT2D_SetWidth,
      s_width.doc(),
      0
    },
    {
      s_shape.name(),
      (getter)PyBobSpIFFT2D_GetShape,
      (setter)PyBobSpIFFT2D_SetShape,
      s_shape.doc(),
      0
    },
    {0}  /* Sentinel */
};


static auto s_transform = bob::extension::FunctionDoc(
  "transform",
  "Calculates the inverse FFT of the given signal",
  ".. note:: The :py:meth:`__call__` function is a synonym for this function.",
  true
)
.add_prototype("input, [output]", "output")
.add_parameter("input", "array_like(2D, complex)", "The given input array. It must be of shape :py:attr:`shape`")
.add_parameter("output", "array_like(2D, complex)", "A pre-allocated output array. If given, it must be of shape :py:attr:`shape`")
.add_return("output", "array_like(2D, complex)", "The inverse FFT result; identical to the ``output`` parameter, if given")
;
static PyObject* PyBobSpIFFT2D_transform(PyBobSpIFFT2DObject* self, PyObject* args, PyObject* kwds) {
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

  if (input->type_num != NPY_COMPLEX128) {
    PyErr_Format(PyExc_TypeError, "`%s' only supports 128-bit complex (2x64-bit float) arrays for input array `input'", Py_TYPE(self)->tp_name);
    return 0;
  }

  if (output && output->type_num != NPY_COMPLEX128) {
    PyErr_Format(PyExc_TypeError, "`%s' only supports 128-bit complex (2x64-bit float) arrays for output array `output'", Py_TYPE(self)->tp_name);
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
    output = (PyBlitzArrayObject*)PyBlitzArray_SimpleNew(NPY_COMPLEX128, 2, size);
    output_ = make_safe(output);
  }

  /** all basic checks are done, can call the operator now **/
  self->cxx->operator()(*PyBlitzArrayCxx_AsBlitz<std::complex<double>,2>(input), *PyBlitzArrayCxx_AsBlitz<std::complex<double>,2>(output));

  return PyBlitzArray_AsNumpyArray(output, 0);
BOB_CATCH_MEMBER("transform", 0)
}

static PyMethodDef PyBobSpIFFT2D_methods[] = {
    {
      s_transform.name(),
      (PyCFunction)PyBobSpIFFT2D_transform,
      METH_VARARGS|METH_KEYWORDS,
      s_transform.doc()
    },
    {0}  /* Sentinel */
};

bool init_BobSpIFFT2D(PyObject* module){
  // class definition
  PyBobSpIFFT2D_Type.tp_name = s_ifft2d.name();
  PyBobSpIFFT2D_Type.tp_basicsize = sizeof(PyBobSpIFFT2DObject);
  PyBobSpIFFT2D_Type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
  PyBobSpIFFT2D_Type.tp_doc = s_ifft2d.doc();

  PyBobSpIFFT2D_Type.tp_new = PyType_GenericNew;
  PyBobSpIFFT2D_Type.tp_dealloc = reinterpret_cast<destructor>(PyBobSpIFFT2D_Delete);
  PyBobSpIFFT2D_Type.tp_init = reinterpret_cast<initproc>(PyBobSpIFFT2D_Init);
  PyBobSpIFFT2D_Type.tp_richcompare = reinterpret_cast<richcmpfunc>(PyBobSpIFFT2D_RichCompare);
  PyBobSpIFFT2D_Type.tp_methods = PyBobSpIFFT2D_methods;
  PyBobSpIFFT2D_Type.tp_getset = PyBobSpIFFT2D_getseters;
  PyBobSpIFFT2D_Type.tp_call = reinterpret_cast<ternaryfunc>(PyBobSpIFFT2D_transform);
  PyBobSpIFFT2D_Type.tp_repr = reinterpret_cast<reprfunc>(PyBobSpIFFT2D_Repr);

  // check that everything is fine
  if (PyType_Ready(&PyBobSpIFFT2D_Type) < 0) return false;
  return PyModule_AddObject(module, "IFFT2D", Py_BuildValue("O", &PyBobSpIFFT2D_Type)) >= 0;
}
