/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Fri 25 Oct 16:54:55 2013
 *
 * @brief Bindings to bob::sp
 */

#define BOB_SP_MODULE
#include <bob.sp/api.h>

int PyBobSp_APIVersion = BOB_SP_API_VERSION;

#ifdef NO_IMPORT_ARRAY
#undef NO_IMPORT_ARRAY
#endif
#include <bob.blitz/capi.h>
#include <bob.blitz/cleanup.h>
#include <bob.core/api.h>

#include "main.h"

static PyMethodDef module_methods[] = {
    {
      s_extrapolate.name(),
      (PyCFunction)PyBobSpExtrapolate,
      METH_VARARGS|METH_KEYWORDS,
      s_extrapolate.doc()
    },
    {
      s_fft.name(),
      (PyCFunction)PyBobSpFFT,
      METH_VARARGS|METH_KEYWORDS,
      s_fft.doc()
    },
    {
      s_ifft.name(),
      (PyCFunction)PyBobSpIFFT,
      METH_VARARGS|METH_KEYWORDS,
      s_ifft.doc()
    },
    {
      s_fftshift.name(),
      (PyCFunction)PyBobSpFFTShift,
      METH_VARARGS|METH_KEYWORDS,
      s_fftshift.doc()
    },
    {
      s_ifftshift.name(),
      (PyCFunction)PyBobSpIFFTShift,
      METH_VARARGS|METH_KEYWORDS,
      s_ifftshift.doc()
    },
    {
      s_dct.name(),
      (PyCFunction)PyBobSpDCT,
      METH_VARARGS|METH_KEYWORDS,
      s_dct.doc()
    },
    {
      s_idct.name(),
      (PyCFunction)PyBobSpIDCT,
      METH_VARARGS|METH_KEYWORDS,
      s_idct.doc()
    },
    {0}  /* Sentinel */
};

PyDoc_STRVAR(module_docstr, "Bob signal processing toolkit");

#if PY_VERSION_HEX >= 0x03000000
static PyModuleDef module_definition = {
  PyModuleDef_HEAD_INIT,
  BOB_EXT_MODULE_NAME,
  module_docstr,
  -1,
  module_methods,
  0, 0, 0, 0
};
#endif

static PyObject* create_module (void) {

# if PY_VERSION_HEX >= 0x03000000
  PyObject* module = PyModule_Create(&module_definition);
  auto module_ = make_xsafe(module);
  const char* ret = "O";
# else
  PyObject* module = Py_InitModule3(BOB_EXT_MODULE_NAME, module_methods, module_docstr);
  const char* ret = "N";
# endif
  if (!module) return 0;

  /* register the types to python */

  if (!init_BobSpFFT1D(module)) return 0;
  if (!init_BobSpIFFT1D(module)) return 0;
  if (!init_BobSpFFT2D(module)) return 0;
  if (!init_BobSpIFFT2D(module)) return 0;

  if (!init_BobSpDCT1D(module)) return 0;
  if (!init_BobSpIDCT1D(module)) return 0;
  if (!init_BobSpDCT2D(module)) return 0;
  if (!init_BobSpIDCT2D(module)) return 0;

  if (!init_BobSpBorderType(module)) return 0;
  if (!init_BobSpQuantization(module)) return 0;

  // initialize the PyBobSp_API
  initialize_api();

#if PY_VERSION_HEX >= 0x02070000

  /* defines the PyCapsule */

  PyObject* c_api_object = PyCapsule_New((void *)PyBobSp_API, BOB_EXT_MODULE_PREFIX "." BOB_EXT_MODULE_NAME "._C_API", 0);

#else

  PyObject* c_api_object = PyCObject_FromVoidPtr((void *)PyBobSp_API, 0);

#endif

  if (!c_api_object) return 0;

  if (PyModule_AddObject(module, "_C_API", c_api_object) < 0) return 0;

  /* imports dependencies */
  if (import_bob_blitz() < 0) return 0;
  if (import_bob_core_logging() < 0) return 0;

  return Py_BuildValue(ret, module);
}

PyMODINIT_FUNC BOB_EXT_ENTRY_NAME (void) {
# if PY_VERSION_HEX >= 0x03000000
  return
# endif
    create_module();
}
