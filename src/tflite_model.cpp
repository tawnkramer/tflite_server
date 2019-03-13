#include "tflite_model.h"
#include <iostream>

using namespace tflite;

#define LOG(x) std::cerr

#define TFLITE_MINIMAL_CHECK(x)                              \
  if (!(x)) {                                                \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
  }

TFLiteModel::TFLiteModel()
{

}
  
TFLiteModel::~TFLiteModel()
{
}

bool TFLiteModel::Load(const char* filename)
{
    m_pModel = FlatBufferModel::BuildFromFile(filename);

    if(m_pModel == nullptr)
    {
        printf("failed to load model:\n %s\n", filename);
        return false;
    }

    // Build the interpreter
    tflite::ops::builtin::BuiltinOpResolver resolver;
    InterpreterBuilder builder(*m_pModel, resolver);
    builder(&m_pInterpreter);
    TFLITE_MINIMAL_CHECK(m_pInterpreter != nullptr);

    // Allocate tensor buffers.
    TFLITE_MINIMAL_CHECK(m_pInterpreter->AllocateTensors() == kTfLiteOk);

    return true;
}

void TFLiteModel::SetNumThreads(int numthreads)
{
    m_pInterpreter->SetNumThreads(numthreads);
}

void TFLiteModel::ShowInputs()
{
	LOG(INFO) << "inputs: " << m_pInterpreter->inputs().size() << "\n";

	int t_size = m_pInterpreter->inputs().size();
	for (int i = 0; i < t_size; i++) 
    {
        int iT = m_pInterpreter->inputs()[i];

		if (m_pInterpreter->tensor(iT)->name)
        {
			LOG(INFO) << i << ": " << m_pInterpreter->tensor(iT)->name << ", "
			<< m_pInterpreter->tensor(iT)->bytes << " bytes, "
			<< "type: " << m_pInterpreter->tensor(iT)->type << ", "
			<< "scale: " << m_pInterpreter->tensor(iT)->params.scale << ", "
			<< "zero_point: " << m_pInterpreter->tensor(iT)->params.zero_point << "\n";

            TfLiteIntArray* dims = m_pInterpreter->tensor(iT)->dims;
            char buf[256];
            sprintf(buf, "(%d, %d, %d)", dims->data[1], dims->data[2], dims->data[3]);
            LOG(INFO) << " dimensions: " << buf << "\n";
        }
	}
}

size_t TFLiteModel::GetInputSize()
{
    size_t total = 0;
    int numInputs = m_pInterpreter->inputs().size();

    for(int i = 0; i < numInputs; i++)
    {
        int iT = m_pInterpreter->inputs()[i];
        total += m_pInterpreter->tensor(iT)->bytes;
    }    

	return total;
}


void TFLiteModel::ShowModel()
{
    LOG(INFO) << "tensors size: " << m_pInterpreter->tensors_size() << "\n";
	LOG(INFO) << "nodes size: " << m_pInterpreter->nodes_size() << "\n";

	int t_size = m_pInterpreter->tensors_size();
	for (int i = 0; i < t_size; i++) {
		if (m_pInterpreter->tensor(i)->name)
			LOG(INFO) << i << ": " << m_pInterpreter->tensor(i)->name << ", "
			<< m_pInterpreter->tensor(i)->bytes << ", "
			<< m_pInterpreter->tensor(i)->type << ", "
			<< m_pInterpreter->tensor(i)->params.scale << ", "
			<< m_pInterpreter->tensor(i)->params.zero_point << "\n";
	}
}

bool TFLiteModel::Inference(void* data, size_t len, std::string& result)
{
    if(len != GetInputSize())
    {
        char buf[1024];
        sprintf(buf, "{ \"err\" : \"ERR >> expected %zu bytes, got %zu\"}", GetInputSize(), len); 
        result = buf;
        return false;
    }

    int i_size = m_pInterpreter->inputs().size();
    char* pData = (char*)data;

	for (int i = 0; i < i_size; i++)
	{
		int iT = m_pInterpreter->inputs()[i];
        size_t numBytes = m_pInterpreter->tensor(iT)->bytes;

		switch (m_pInterpreter->tensor(iT)->type) 
        {            
            case kTfLiteFloat32:
                memcpy(m_pInterpreter->typed_tensor<float>(iT), pData, numBytes);
                break;
            
            case kTfLiteUInt8:
                memcpy(m_pInterpreter->typed_tensor<uint8_t>(iT), pData, numBytes);
                break;
            
            default:
                LOG(FATAL) << "cannot handle input type "
                            << m_pInterpreter->tensor(iT)->type << " yet";
                result = "{ \"err\" : \"ERR >> can't handle tensorf input type.\"}";
                return false;
        }

        //advance to next input
        pData += numBytes;
    }

	// Invoke inference
    TFLITE_MINIMAL_CHECK(m_pInterpreter->Invoke() == kTfLiteOk);
    
    return true;
}

void TFLiteModel::GetResultJson(std::string& result)
{
    char buf[1024];
    sprintf(buf, "{ \"err\" : \"none\", \"result\" : [ "); 
    result = buf;

    int o_size = m_pInterpreter->inputs().size();
    
    for(int iOut = 0; iOut < o_size; iOut++)
    {
        int output = m_pInterpreter->outputs()[iOut];
        
        TfLiteIntArray* output_dims = m_pInterpreter->tensor(output)->dims;
        
        // assume output dims to be something like (1, 1, ... ,size)
        auto output_size = output_dims->data[output_dims->size - 1];

        switch (m_pInterpreter->tensor(output)->type) {
            case kTfLiteFloat32:
                {
                    float* pRes = m_pInterpreter->typed_output_tensor<float>(iOut);
                    result += "[ ";
                    
                    for(int iRes = 0; iRes < output_size; iRes++)
                    {
                        if(iRes == output_size - 1)
                            sprintf(buf, "%f", *pRes);
                        else
                            sprintf(buf, "%f, ", *pRes);

                        result += buf;
                        pRes++;
                    }

                    result += " ]";
                }
                break;
            case kTfLiteUInt8:
                {
                    uint8_t* pRes = m_pInterpreter->typed_output_tensor<uint8_t>(iOut);
                    result += "[ ";
                    
                    for(int iRes = 0; iRes < output_size; iRes++)
                    {
                        if(iRes == output_size - 1)
                            sprintf(buf, "%d", *pRes);
                        else
                            sprintf(buf, "%d, ", *pRes);
                            
                        result += buf;
                        pRes++;
                    }

                    result += " ]";           
                }
                break;
            default:
                LOG(FATAL) << "cannot handle output type "
                            << m_pInterpreter->tensor(output)->type << " yet";
                exit(-1);
        }

        if(iOut < o_size - 1)
        {
            result += ", ";
        }

    }
    
    result += " ] }";
    
}