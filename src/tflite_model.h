#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include <string>

class TFLiteModel
{
public:
  TFLiteModel();
  ~TFLiteModel();
  bool Load(const char* filename);
  void ShowInputs();
  void ShowModel();
  void SetNumThreads(int numthreads);
  size_t GetInputSize();
  bool Inference(void* data, size_t len, std::string& result);
  void GetResultJson(std::string& result);

  std::unique_ptr<tflite::FlatBufferModel> m_pModel;
  std::unique_ptr<tflite::Interpreter> m_pInterpreter;
};

