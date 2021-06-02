#include <stdio.h>
#include <limits>
#include "itkCommand.h"
#include "itkImageFileWriter.h"
#include "itkAmoebaOptimizerv4.h"
#include "itkRegularStepGradientDescentOptimizerv4.h"
#include "itkMeanSquaresImageToImageMetricv4.h"

#define PLM_CONFIG_ITKV4_REGISTRATION 1
#include "itk_image_load.h"
#include "itk_registration_private.h"
#include "xform.h"

//#define USE_RSG 1

#if USE_RSG
using OptimizerType = itk::RegularStepGradientDescentOptimizerv4<double>;
#else
using OptimizerType = itk::AmoebaOptimizerv4;
#endif

//using OptimizerType = itk::OnePlusOneEvolutionaryOptimizerv4<double>;
using OptimizerPointer = const OptimizerType *;

using MetricType = itk::ObjectToObjectMetricBaseTemplate< double >;
using ImageMetricType = itk::ImageToImageMetricv4<
    FloatImageType, FloatImageType, FloatImageType, double>;
using MSEMetricType = itk::MeanSquaresImageToImageMetricv4<
    FloatImageType, FloatImageType >;
typedef itk::LinearInterpolateImageFunction <
    FloatImageType, double >InterpolatorType;

class CommandIterationUpdate : public itk::Command
{
public:
  using Self = CommandIterationUpdate;
  using Superclass = itk::Command;
  using Pointer = itk::SmartPointer<Self>;
  itkNewMacro(Self);

protected:
  CommandIterationUpdate() { m_LastMetricValue = 0.0; };

public:

  void
  Execute(itk::Object * caller, const itk::EventObject & event) override
  {
    Execute((const itk::Object *)caller, event);
  }

  void
  Execute(const itk::Object * object, const itk::EventObject & event) override
  {
    auto optimizer = static_cast<OptimizerPointer>(object);
    if (!itk::IterationEvent().CheckEvent(&event))
    {
      return;
    }
    double currentValue = optimizer->GetValue();
    // Only print out when the Metric value changes
    if (std::fabs(m_LastMetricValue - currentValue) > 1e-7)
    {
      std::cout << optimizer->GetCurrentIteration() << "   ";
      std::cout << currentValue << "   ";
      std::cout << optimizer->GetCurrentPosition() << std::endl;
      m_LastMetricValue = currentValue;
    }
  }

private:
  double m_LastMetricValue;
};


int main 
(
    int argc,
    char* argv[]
)
{
    FloatImageType::Pointer fixed = itk_image_load_float (
        "/PHShome/gcs6/build/plastimatch-itkv4/Testing/gauss-1.mha", 0);
    FloatImageType::Pointer moving = itk_image_load_float (
        "/PHShome/gcs6/build/plastimatch-itkv4/Testing/gauss-2.mha", 0);

    RegistrationType::Pointer registration = RegistrationType::New();
    registration->SetFixedImage (fixed);
    registration->SetMovingImage (moving);

    OptimizerType::Pointer optimizer = OptimizerType::New();
#if USE_RSG
    optimizer->SetLearningRate(15);
#endif
    registration->SetOptimizer (optimizer);

    CommandIterationUpdate::Pointer observer = CommandIterationUpdate::New();
    optimizer->AddObserver(itk::IterationEvent(), observer);

    MSEMetricType::Pointer metric = MSEMetricType::New();
    registration->SetMetric (metric);

    InterpolatorType::Pointer interpolator = InterpolatorType::New();
    metric->SetFixedInterpolator (interpolator);
    InterpolatorType::Pointer interpolator2 = InterpolatorType::New();
    metric->SetMovingInterpolator (interpolator2);

    TranslationTransformType::Pointer transform =TranslationTransformType::New();
    registration->SetInitialTransform (transform);
    std::cout << "Initial Parameters = " 
        << registration->GetTransform()->GetParameters() << "\n";

    registration->Update ();
}
