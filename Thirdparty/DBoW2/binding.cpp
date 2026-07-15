#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "DBoW2/BowVector.h"
#include "DBoW2/FORB.h"
#include "DBoW2/TemplatedVocabulary.h"

namespace py = pybind11;
using OrbVocabulary = DBoW2::TemplatedVocabulary<DBoW2::FORB::TDescriptor, DBoW2::FORB>;

namespace {

std::vector<cv::Mat> descriptor_rows(
    const py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast>& descriptors) {
  const py::buffer_info info = descriptors.request();
  if (info.ndim != 2 || info.shape[1] != DBoW2::FORB::L) {
    throw std::invalid_argument("ORB descriptors must have shape (N, 32)");
  }
  auto* data = static_cast<std::uint8_t*>(info.ptr);
  std::vector<cv::Mat> rows;
  rows.reserve(static_cast<std::size_t>(info.shape[0]));
  for (py::ssize_t row = 0; row < info.shape[0]; ++row) {
    rows.emplace_back(1, DBoW2::FORB::L, CV_8U, data + row * info.shape[1]);
  }
  return rows;
}

class OrbDatabase {
 public:
  explicit OrbDatabase(const std::string& vocabulary_path)
      : vocabulary_(std::make_unique<OrbVocabulary>()) {
    std::ifstream vocabulary_stream(vocabulary_path);
    if (!vocabulary_stream.good()) {
      throw std::runtime_error("cannot open ORB vocabulary: " + vocabulary_path);
    }
    vocabulary_stream.close();
    if (!vocabulary_->loadFromTextFile(vocabulary_path)) {
      throw std::runtime_error("failed to load ORB vocabulary: " + vocabulary_path);
    }
  }

  std::size_t size() const { return entries_.size(); }

  py::dict metadata() const {
    py::dict result;
    result["branching"] = vocabulary_->getBranchingFactor();
    result["depth"] = vocabulary_->getDepthLevels();
    result["weighting"] = static_cast<int>(vocabulary_->getWeightingType());
    result["scoring"] = static_cast<int>(vocabulary_->getScoringType());
    result["words"] = vocabulary_->size();
    return result;
  }

  std::size_t add(
      const py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast>& descriptors) {
    DBoW2::BowVector bow;
    transform(descriptors, bow);
    const std::size_t entry_id = entries_.size();
    entries_.push_back(std::move(bow));
    return entry_id;
  }

  std::vector<std::pair<std::size_t, double>> query(
      const py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast>& descriptors) const {
    DBoW2::BowVector query_bow;
    transform(descriptors, query_bow);
    std::vector<std::pair<std::size_t, double>> results;
    results.reserve(entries_.size());
    for (std::size_t entry_id = 0; entry_id < entries_.size(); ++entry_id) {
      const double score = vocabulary_->score(query_bow, entries_[entry_id]);
      // The ORB-SLAM DBoW2 fork has no database class.  Preserve its vocabulary
      // score semantics and omit entries with no shared weighted words.
      if (std::isfinite(score) && score > 0.0) {
        results.emplace_back(entry_id, score);
      }
    }
    std::sort(results.begin(), results.end(), [](const auto& lhs, const auto& rhs) {
      if (lhs.second != rhs.second) return lhs.second > rhs.second;
      return lhs.first < rhs.first;
    });
    return results;
  }

 private:
  void transform(
      const py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast>& descriptors,
      DBoW2::BowVector& bow) const {
    const auto rows = descriptor_rows(descriptors);
    if (!rows.empty()) vocabulary_->transform(rows, bow);
  }

  std::unique_ptr<OrbVocabulary> vocabulary_;
  std::vector<DBoW2::BowVector> entries_;
};

}  // namespace

PYBIND11_MODULE(macvo_dbow2, module) {
  module.doc() = "Minimal ORB-SLAM3 DBoW2 binding for MAC-VO place recognition";
  module.attr("orb_slam3_commit") = "0df83dde1c85c7ab91a0d47de7a29685d046f637";
  py::class_<OrbDatabase>(module, "OrbDatabase")
      .def(py::init<const std::string&>())
      .def("size", &OrbDatabase::size)
      .def("metadata", &OrbDatabase::metadata)
      .def("add", &OrbDatabase::add)
      .def("query", &OrbDatabase::query);
}
