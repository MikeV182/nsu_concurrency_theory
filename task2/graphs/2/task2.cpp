#include <vector>

#include "pbPlots.hpp"
#include "supportLib.hpp"


void render_graph(std::vector<double> &x, std::vector<double> &y, std::string &filename) {
    ScatterPlotSeries *series = GetDefaultScatterPlotSeriesSettings();
	series->xs = &x;
	series->ys = &y;
	series->linearInterpolation = true;
	series->lineType = toVector(L"solid");
    series->pointType = toVector(L"dots");
	series->lineThickness = 1;

    ScatterPlotSettings *settings = GetDefaultScatterPlotSettings();
    settings->width = 800;
	settings->height = 600;
    settings->autoBoundaries = true;
	settings->autoPadding = true;
    settings->xLabel = toVector(L"P");
	settings->yLabel = toVector(L"Sp");
    settings->scatterPlotSeries->push_back(series);

    RGBABitmapImageReference *imageRef = CreateRGBABitmapImageReference();
    StringReference *errorMessage = CreateStringReferenceLengthValue(0, L' ');

    bool result = DrawScatterPlotFromSettings(imageRef, settings, errorMessage);

    if (result) {
        std::vector<double> *pngData = ConvertToPNG(imageRef->image);
        WriteToFile(pngData, filename);
        DeleteImage(imageRef->image);
    } else {
        std::cerr << "Error: ";
		for(wchar_t c : *errorMessage->string){
			std::wcerr << c;
		}
		std::cerr << std::endl;
    }

    FreeAllocations();
}

int main() {
    std::vector<double> num_threads{1, 2, 4, 7, 8, 16, 20, 40};
    std::vector <double> y{1.00, 1.79, 3.36, 5.20, 5.73, 9.42, 9.97, 13.02};
    std::string task2_filename = "./task2.png";
    render_graph(num_threads, y, task2_filename);
}