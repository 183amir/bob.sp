/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Thu 13 Feb 2014 15:34:30 CET
 *
 * @brief Methods for quick FFT/IFFT calculation
 */

#include "main.h"

#include <bob.sp/fftshift.h>

static PyBlitzArrayObject* check_and_allocate(PyBlitzArrayObject* input, PyBlitzArrayObject* output) {
  if (input->type_num != NPY_COMPLEX128) {
    PyErr_SetString(PyExc_TypeError, "method only supports 128-bit complex (2x64-bit float) arrays for input array `input'");
    return 0;
  }

  if (output && output->type_num != NPY_COMPLEX128) {
    PyErr_SetString(PyExc_TypeError, "method only supports 128-bit complex (2x64-bit float) arrays for output array `output'");
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
    output = (PyBlitzArrayObject*)PyBlitzArray_SimpleNew(NPY_COMPLEX128, input->ndim, input->shape);
  }

  return output;
}


bob::extension::FunctionDoc s_fft = bob::extension::FunctionDoc(
  "fft",
  "Computes the direct Fast Fourier Transform of a 1D or 2D array/signal of type ``complex128``",
  "Allocates a new output array if ``dst`` is not provided. If it is, then it must be of the same type and shape as ``src``."
)
.add_prototype("src, [dst]", "dst")
.add_parameter("src", "array_like(1D or 2D, complex)", "A 1 or 2-dimensional array of type ``complex128`` for which the FFT operation will be performed")
.add_parameter("dst", "array_like(1D or 2D, complex)", "A 1 or 2-dimensional array of type ``complex128`` and matching dimensions to ``src`` in  which the result of the operation will be stored")
.add_return("dst", "array_like(1D or 2D, complex)", "The 1 or 2-dimensional array of type ``complex128`` of the same dimension as ``src``, containing the FFT of the ``src`` input signal")
;
PyObject* PyBobSpFFT(PyObject*, PyObject* args, PyObject* kwds) {
BOB_TRY
  char** kwlist = s_fft.kwlist();

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
    bob::sp::FFT1D op(input->shape[0]);
    op(*PyBlitzArrayCxx_AsBlitz<std::complex<double>,1>(input), *PyBlitzArrayCxx_AsBlitz<std::complex<double>,1>(output));
  } else { // input->ndim == 2
    bob::sp::FFT2D op(input->shape[0], input->shape[1]);
    op(*PyBlitzArrayCxx_AsBlitz<std::complex<double>,2>(input), *PyBlitzArrayCxx_AsBlitz<std::complex<double>,2>(output));
  }
  return PyBlitzArray_AsNumpyArray(output, 0);
BOB_CATCH_FUNCTION("fft", 0)
}


bob::extension::FunctionDoc s_ifft = bob::extension::FunctionDoc(
  "ifft",
  "Computes the inverse Fast Fourier Transform of a 1D or 2D array/signal of type ``complex128``",
  "Allocates a new output array if ``dst`` is not provided. If it is, then it must be of the same type and shape as ``src``."
)
.add_prototype("src, [dst]", "dst")
.add_parameter("src", "array_like(1D or 2D, complex)", "A 1 or 2-dimensional array of type ``complex128`` for which the inverse DCT operation will be performed")
.add_parameter("dst", "array_like(1D or 2D, complex)", "A 1 or 2-dimensional array of type ``complex128`` and matching dimensions to ``src`` in  which the result of the operation will be stored")
.add_return("dst", "array_like(1D or 2D, complex)", "The 1 or 2-dimensional array of type ``complex128`` of the same dimension as ``src``, containing the inverse FFT of the ``src`` input signal")
;
PyObject* PyBobSpIFFT(PyObject*, PyObject* args, PyObject* kwds) {
BOB_TRY
  char** kwlist = s_ifft.kwlist();

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
    bob::sp::IFFT1D op(input->shape[0]);
    op(*PyBlitzArrayCxx_AsBlitz<std::complex<double>,1>(input), *PyBlitzArrayCxx_AsBlitz<std::complex<double>,1>(output));
  }

  else { // input->ndim == 2
    bob::sp::IFFT2D op(input->shape[0], input->shape[1]);
    op(*PyBlitzArrayCxx_AsBlitz<std::complex<double>,2>(input),*PyBlitzArrayCxx_AsBlitz<std::complex<double>,2>(output));
  }
  return PyBlitzArray_AsNumpyArray(output, 0);
BOB_CATCH_FUNCTION("ifft", 0)
}


bob::extension::FunctionDoc s_fftshift = bob::extension::FunctionDoc(
  "fftshift",
  "Shifts the given data such that the center of the data is centered at zero for FFT, as required by the FFT",
  "If a 1D ``complex128`` array is passed, inverses the two halves of that array and returns the result as a new array. "
  "If a 2D ``complex128`` array is passed, swaps the four quadrants of the array and returns the result as a new array."
)
.add_prototype("src, [dst]", "dst")
.add_parameter("src", "array_like(1D or 2D, complex)", "A 1 or 2-dimensional array of type ``complex128`` to be shifted")
.add_parameter("dst", "array_like(1D or 2D, complex)", "A  pre-allocated 1 or 2-dimensional array of type ``complex128`` and matching dimensions to ``src`` in  which the result of the operation will be stored")
.add_return("dst", "array_like(1D or 2D, complex)", "The 1 or 2-dimensional array of type ``complex128`` of the same dimension as ``src``, containing the shifted version of ``src``")
;
PyObject* PyBobSpFFTShift(PyObject*, PyObject* args, PyObject* kwds) {
BOB_TRY
  char** kwlist = s_fftshift.kwlist();

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
    bob::sp::fftshift(
        *PyBlitzArrayCxx_AsBlitz<std::complex<double>,1>(input),
        *PyBlitzArrayCxx_AsBlitz<std::complex<double>,1>(output)
        );
  }

  else { // input->ndim == 2
    bob::sp::fftshift(
        *PyBlitzArrayCxx_AsBlitz<std::complex<double>,2>(input),
        *PyBlitzArrayCxx_AsBlitz<std::complex<double>,2>(output)
        );
  }
  return PyBlitzArray_AsNumpyArray(output, 0);
BOB_CATCH_FUNCTION("fftshift", 0)
}


bob::extension::FunctionDoc s_ifftshift = bob::extension::FunctionDoc(
  "ifftshift",
  "This method undoes what :py:meth:`fftshift` does",
  "It accepts 1 or 2-dimensional arrays of type ``complex128``"
)
.add_prototype("src, [dst]", "dst")
.add_parameter("src", "array_like(1D or 2D, complex)", "A 1 or 2-dimensional array of type ``complex128`` to be shifted back")
.add_parameter("dst", "array_like(1D or 2D, complex)", "A  pre-allocated 1 or 2-dimensional array of type ``complex128`` and matching dimensions to ``src`` in  which the result of the operation will be stored")
.add_return("dst", "array_like(1D or 2D, complex)", "The 1 or 2-dimensional array of type ``complex128`` of the same dimension as ``src``, containing the back-shifted version of ``src``")
;
PyObject* PyBobSpIFFTShift(PyObject*, PyObject* args, PyObject* kwds) {
BOB_TRY
  char** kwlist = s_ifftshift.kwlist();

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
    bob::sp::ifftshift(
        *PyBlitzArrayCxx_AsBlitz<std::complex<double>,1>(input),
        *PyBlitzArrayCxx_AsBlitz<std::complex<double>,1>(output)
        );
  }

  else { // input->ndim == 2
    bob::sp::ifftshift(
        *PyBlitzArrayCxx_AsBlitz<std::complex<double>,2>(input),
        *PyBlitzArrayCxx_AsBlitz<std::complex<double>,2>(output)
        );
  }
  return PyBlitzArray_AsNumpyArray(output, 0);
BOB_CATCH_FUNCTION("fftshift", 0)
}
