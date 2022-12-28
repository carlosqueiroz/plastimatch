using System;
namespace Application
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Hello, World!");
            var fn1 = "C:/Users/gcs6/build/x64/plastimatch/Testing/gauss-2.mha";
            var fn2 = "C:/Users/gcs6/build/x64/plastimatch/Testing/gauss-3.mha";
            var pi1 = Plm_image.New();
            pi1.load_native(fn1);
            var pi2 = Plm_image.New();
            pi2.load_native(fn2);
            var gdc = new Gamma_dose_comparison();
            gdc.set_reference_image(pi1);
        }
    }
}
