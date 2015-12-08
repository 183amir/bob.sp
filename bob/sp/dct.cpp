/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Thu 13 Feb 2014 16:52:04 CET
 *
 * @brief Methods for quick DCT/IDCT calculation
 */

#include "main.h"

static PyBlitzArrayObject* check_and_allocate(PyBlitzArrayObject* input, PyBlitzArrayObject* output) {

  if (input->type_num != NPY_FLOAT64) {
    PyErr_SetString(PyExc_TypeError, "method only supports 64-bit float arrays for input array `input'");
    return 0;
  }

  if (output && output->type_num != NPY_FLOAT64) {
    PyErr_SetString(PyExc_TypeError, "method only supports 64-bit float arrays for output array `output'");
    return 0;
  }

  if (input->ndim != 1 and input->ndim != 2) {
    PyErr_Format(PyExc_TypeError, "method only accepts 1 or 2-dimensional arrays (not %" PY_FORMAT_SIZE_T "dD arrays)", input->ndim);
    return 0;
  }

  if (output && input->ndim != output->ndim) {
    PyErr_Format(PyExc_RuntimeError, "input and output arrays should have matching number of dimensions, but input array `input' has %" PY_FORMAT_SIZE_T "d dimensions while output array `output' has %" PY_FORMAT_SIZE_T "d dimensions", input->ndim, output->ndim);
    return 0;
  }

  if (output) {
    if (input->ndim == 1) {
      if (output->shape[0] != input->shape[0]) {
        PyErr_Format(PyExc_RuntimeError, "1D `output' array should have %" PY_FORMAT_SIZE_T "d elements matching output size, not %" PY_FORMAT_SIZE_T "d elements", input->shape[0], output->shape[0]);
        return 0;
      }
    }
    else { // input->ndim == 2
      if (output->shape[0] != input->shape[0]) {
        PyErr_Format(PyExc_RuntimeError, "2D `output' array should have %" PY_FORMAT_SIZE_T "d rows matching input size, not %" PY_FORMAT_SIZE_T "d rows", input->shape[0], output->shape[0]);
        return 0;
      }
      if (output->shape[1] != input->shape[1]) {
        PyErr_Format(PyExc_RuntimeError, "2D `output' array should have %" PY_FORMAT_SIZE_T "d columns matching input size, not %" PY_FORMAT_SIZE_T "d columns", input->shape[1], output->shape[1]);
        return 0;
      }
    }
    Py_INCREF(output);
  } else {
    output = (PyBlitzArrayObject*)PyBlitzArray_SimpleNew(NPY_FLOAT64, input->ndim, input->shape);
  }
  return output;
}


bob::extension::FunctionDoc s_dct = bob::extension::FunctionDoc(
  "dct",
  "Computes the direct Discrete Cosine Transform of a 1D or 2D array/signal of type ``float64``",
  "Allocates a new output array if ``dst`` is not provided. If it is, then it must be of the same type and shape as ``src``."
)
.add_prototype("src, [dst]", "dst")
.add_parameter("src", "array_like(1D or 2D, float)", "A 1 or 2-dimensional array of type ``float64`` for which the DCT operation will be performed")
.add_parameter("dst", "array_like(1D or 2D, float)", "A 1 or 2-dimensional array of type ``float64`` and matching dimensions to ``src`` in  which the result of the operation will be stored")
.add_return("dst", "array_like(1D or 2D, float)", "The 1 or 2-dimensional array of type ``float64`` of the same dimension as ``src`` and of type ``float64``, containing the DCT of the ``src`` input signal")
;
PyObject* PyBobSpDCT(PyObject*, PyObject* args, PyObject* kwds) {
BOB_TRY
  char** kwlist = s_dct.kwlist();

  PyBlitzArrayObject* input;
  PyBlitzArrayObject* output = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&|O&", kwlist,
        &PyBlitzArray_Converter, &input,
        &PyBlitzArray_OutputConverter, &output
        )) return 0;

  //protects acquired resources through this scope
  auto input_ = make_safe(input);
  auto output_ = make_xsafe(output);

  output = check_and_allocate(input, output);
  if (!output) return 0;
  output_ = make_safe(output);

  /** all basic checks are done, can call the operator now **/
  if (input->ndim == 1) {
    bob::sp::DCT1D op(input->shape[0]);
    op(*PyBlitzArrayCxx_AsBlitz<double,1>(input), *PyBlitzArrayCxx_AsBlitz<double,1>(output));
  }

  else { // input->ndim == 2
    bob::sp::DCT2D op(input->shape[0], input->shape[1]);
    op(*PyBlitzArrayCxx_AsBlitz<double,2>(input), *PyBlitzArrayCxx_AsBlitz<double,2>(output));
  }
  return PyBlitzArray_AsNumpyArray(output, 0);
BOB_CATCH_FUNCTION("dct", 0)
}


bob::extension::FunctionDoc s_idct = bob::extension::FunctionDoc(
  "idct",
  "Computes the inverse Discrete Cosinte Transform of a 1D or 2D array/signal of type ``float64``",
  "Allocates a new output array if ``dst`` is not provided. If it is, then it must be of the same type and shape as ``src``."
)
.add_prototype("src, [dst]", "dst")
.add_parameter("src", "array_like(1D or 2D, float)", "A 1 or 2-dimensional array of type ``float64`` for which the inverse DCT operation will be performed")
.add_parameter("dst", "array_like(1D or 2D, float)", "A 1 or 2-dimensional array of type ``float64`` and matching dimensions to ``src`` in  which the result of the operation will be stored")
.add_return("dst", "array_like(1D or 2D, float)", "The 1 or 2-dimensional array of type ``float64`` of the same dimension as ``src`` and of type ``float64``, containing the inverse DCT of the ``src`` input signal")
;
PyObject* PyBobSpIDCT(PyObject*, PyObject* args, PyObject* kwds) {
BOB_TRY
  char** kwlist = s_idct.kwlist();

  PyBlitzArrayObject* input;
  PyBlitzArrayObject* output = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&|O&", kwlist,
        &PyBlitzArray_Converter, &input,
        &PyBlitzArray_OutputConverter, &output
        )) return 0;

  //protects acquired resources through this scope
  auto input_ = make_safe(input);
  auto output_ = make_xsafe(output);

  output = check_and_allocate(input, output);
  if (!output) return 0;
  output_ = make_safe(output);

  /** all basic checks are done, can call the operator now **/
  if (input->ndim == 1) {
    bob::sp::IDCT1D op(input->shape[0]);
    op(*PyBlitzArrayCxx_AsBlitz<double,1>(input), *PyBlitzArrayCxx_AsBlitz<double,1>(output));
  } else { // input->ndim == 2
    bob::sp::IDCT2D op(input->shape[0], input->shape[1]);
    op(*PyBlitzArrayCxx_AsBlitz<double,2>(input), *PyBlitzArrayCxx_AsBlitz<double,2>(output));
  }
  return PyBlitzArray_AsNumpyArray(output, 0);
BOB_CATCH_FUNCTION("idct", 0)
}
