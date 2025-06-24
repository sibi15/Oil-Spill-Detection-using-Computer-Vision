import React, { useState } from "react";
import { Upload, AlertCircle, Ruler, Lightbulb, Activity } from "lucide-react";
import axios from "axios";
import { imageTypes, ImageType } from "../utils/imageTypes";
import { config } from "../config";

interface AnalysisResult {
  processedImageUrl: string;
  densityGraph?: string;
  metrics: {
    iou: number;
    union: number;
    intersection: number;
    spill_area: number;
    mean_intensity: number;
    standard_deviation: number;
    spill_pixels: number;
  };
}

const ImageUploadSection: React.FC = () => {
  const [selectedType, setSelectedType] = useState<string>(imageTypes[0].id);
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState<boolean>(false);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(
    null
  );
  const [error, setError] = useState<string | null>(null);

  const handleTypeChange = (type: string) => {
    setSelectedType(type);
    setAnalysisResult(null);
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setError(null);
    setAnalysisResult(null);

    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0];

      if (!selectedFile.type.startsWith("image/")) {
        setError("Please upload an image file");
        return;
      }

      setFile(selectedFile);
      setPreviewUrl(URL.createObjectURL(selectedFile));
    }
  };

  const handleUpload = async () => {
    if (!file) {
      setError("Please select a file first");
      return;
    }

    setIsUploading(true);
    setError(null);

    const formData = new FormData();
    formData.append("file", file);
    formData.append("imageType", selectedType);

    try {
      const response = await axios.post(
        `${config.apiUrl}/predict`,
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        }
      );

      const { processed_image, density_graph, metrics } = response.data;
      setAnalysisResult({
        processedImageUrl: `data:image/png;base64,${processed_image}`,
        densityGraph: density_graph,
        metrics: {
          iou: metrics.iou,
          union: metrics.union,
          intersection: metrics.intersection,
          spill_area: metrics.spill_area,
          mean_intensity: metrics.mean_intensity,
          standard_deviation: metrics.standard_deviation,
          spill_pixels: metrics.spill_pixels,
        },
      });
    } catch (err: any) {
      // log detailed error to console, don't display in UI
      console.error("Upload error:", err.response?.data?.error || err);
    } finally {
      setIsUploading(false);
    }
  };

  const MetricCard = ({
    icon: Icon,
    label,
    value,
    color,
  }: {
    icon: any;
    label: string;
    value: number | string;
    color: string;
  }) => (
    <div className="bg-white rounded-lg shadow-md p-4 flex flex-col items-center justify-center text-center">
      <div
        className={`${color} rounded-full w-10 h-10 flex items-center justify-center mb-3`}
      >
        <Icon className="h-6 w-6 text-white" />
      </div>
      <p className="text-sm text-gray-600 mb-1">{label}</p>
      <p className="text-lg font-semibold text-blue-900">
        {typeof value === "number" ? value.toFixed(4) : value}
      </p>
    </div>
  );

  const selectedTypeData = imageTypes.find((type) => type.id === selectedType);

  return (
    <section id="upload" className="py-12 px-6 bg-white">
      <div className="container mx-auto">
        <h2 className="text-3xl font-bold mb-8 text-center text-blue-900">
          Image Upload and Processing
        </h2>

        <div className="max-w-6xl mx-auto bg-gray-50 rounded-lg shadow-lg p-6">
          <h3 className="text-xl font-semibold mb-6 text-blue-800">
            Analysis Method
          </h3>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-8">
            {imageTypes.map((type) => {
              const Icon = type.icon;
              return (
                <button
                  key={type.id}
                  className={`relative p-4 rounded-lg border-2 transition-all duration-200 flex flex-col items-center text-center ${
                    selectedType === type.id
                      ? "border-blue-600 bg-blue-50 text-blue-800"
                      : "border-gray-300 hover:border-blue-400 text-gray-700"
                  }`}
                  onClick={() => handleTypeChange(type.id)}
                >
                  <Icon className="h-8 w-8 mb-2 text-blue-600" />
                  <h4 className="font-semibold mb-1">{type.name}</h4>
                  <p className="text-sm text-gray-600">{type.description}</p>
                </button>
              );
            })}
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            <div>
              <h3 className="text-xl font-semibold mb-4 text-blue-800">
                Upload Image
              </h3>

              <div className="mb-6">
                <div
                  className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center cursor-pointer hover:border-blue-500 transition-colors duration-200"
                  onClick={() => document.getElementById("fileInput")?.click()}
                >
                  {previewUrl ? (
                    <div className="flex flex-col items-center">
                      <img
                        src={previewUrl}
                        alt="Preview"
                        className="max-h-60 max-w-full mb-4 rounded"
                      />
                      <p className="text-sm text-gray-500">
                        Click to change image
                      </p>
                    </div>
                  ) : (
                    <div className="flex flex-col items-center">
                      <Upload className="h-12 w-12 text-blue-500 mb-2" />
                      <p className="text-blue-600 font-medium">
                        {selectedTypeData?.uploadText || "Upload Image"}
                      </p>
                      <p className="text-sm text-gray-500 mt-1">
                        or drag and drop
                      </p>
                    </div>
                  )}
                  <input
                    id="fileInput"
                    type="file"
                    className="hidden"
                    accept="image/*"
                    onChange={handleFileChange}
                  />
                </div>

                {error && (
                  <div className="mt-4 p-3 bg-red-50 text-red-700 rounded-lg flex items-center">
                    <AlertCircle className="h-5 w-5 mr-2" />
                    {error}
                  </div>
                )}
              </div>

              <button
                className={`w-full py-3 px-4 rounded-lg text-white font-medium transition-all duration-200 ${
                  file && !isUploading
                    ? "bg-blue-600 hover:bg-blue-700"
                    : "bg-gray-400 cursor-not-allowed"
                }`}
                onClick={handleUpload}
                disabled={!file || isUploading}
              >
                {isUploading ? "Analyzing..." : "Analyze Image"}
              </button>
            </div>

            {analysisResult && (
              <div className="col-span-1 lg:col-span-2 bg-white rounded-lg shadow-md p-6">
                <h4 className="text-lg font-semibold text-blue-800 mb-4">
                  Processed Image
                </h4>
                <img
                  src={analysisResult.processedImageUrl}
                  alt="Processed"
                  className="w-full rounded-lg mb-6"
                />
                {analysisResult.densityGraph && (
                  <>
                    <h4 className="text-lg font-semibold text-blue-800 mb-4 mt-6">
                      Density Graph
                    </h4>
                    <img
                      src={`data:image/png;base64,${analysisResult.densityGraph}`}
                      alt="Density Graph"
                      className="w-full rounded-lg mb-6"
                    />
                  </>
                )}
                <div className="flex flex-wrap justify-evenly items-stretch w-full">
                  {analysisResult.metrics.mean_intensity > 0 && (
                    <MetricCard
                      icon={Lightbulb}
                      label="Mean Intensity"
                      value={`${analysisResult.metrics.mean_intensity.toFixed(2)}`}
                      color="bg-blue-500"
                    />
                  )}
                  {analysisResult.metrics.standard_deviation > 0 && (
                    <MetricCard
                      icon={AlertCircle}
                      label="Standard Deviation"
                      value={`${analysisResult.metrics.standard_deviation.toFixed(2)}`}
                      color="bg-red-500"
                    />
                  )}
                  {analysisResult.metrics.spill_pixels > 0 && (
                    <MetricCard
                      icon={Ruler}
                      label="Total Oil Spill Pixels"
                      value={`${analysisResult.metrics.spill_pixels.toFixed(0)}`}
                      color="bg-indigo-500"
                    />
                  )}
                  {analysisResult.metrics.spill_area > 0 && (
                    <MetricCard
                      icon={Activity}
                      label="Total Spill Area"
                      value={`${analysisResult.metrics.spill_area.toFixed(0)}%`}
                      color="bg-green-500"
                    />
                  )}
                  {/*
                  {analysisResult.metrics.iou >= 0 && (
                    <MetricCard icon={Activity} label="Intersection over Union (IoU)" value={analysisResult.metrics.iou} color="bg-green-500" />
                  )}
                  {analysisResult.metrics.intersection >= 0 && (
                    <MetricCard icon={Activity} label="Intersection" value={analysisResult.metrics.intersection} color="bg-green-500" />
                  )}
                  {analysisResult.metrics.union >= 0 && (
                    <MetricCard icon={Activity} label="Union" value={analysisResult.metrics.union} color="bg-green-500" />
                  )}
                  */}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </section>
  );
};

export default ImageUploadSection;
