# Báo cáo bài thực hành Ngày 1 – Đọc nhãn từ đầu ra YOLO11

**Ngày chạy:** 11/09/2026

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> ZIP do notebook tạo có tên `KX-DAY01-report.zip`. Giải nén rồi đặt trực tiếp `REPORT.md` và
> `day1_lab_outputs/` vào thư mục `report/` của repository tạo từ template. Không ghi họ tên, MSSV,
> email, số điện thoại hoặc dữ liệu cá nhân khác. Nộp link repository trên VLearn; tài khoản VLearn xác
> định người nộp.

## 1. Phân loại ảnh – prediction cấp ảnh

Nguồn evidence: `classification_predictions.json`, sample `traffic`.

- Record hạng 1 (`class_id`, `class_name`, `rank`, `score`, `taxonomy_name`):
`class_id`: `468`,
`class_name`: `cab`,
`rank`: `1`,
`score`:` 0.510915`,
`taxonomy_name`:  `ImageNet-1K"
- Record này mô tả toàn ảnh như thế nào? 
Đây là prediction cấp ảnh: checkpoint xếp toàn bộ ảnh `traffic` gần với lớp `cab` (taxi) ở hạng 1, với score 0.510915. Nhãn này mô tả nội dung nổi bật mà mô hình suy ra cho toàn ảnh; nó không cho biết vị trí, số lượng hay từng vật thể trong ảnh.
- Ai định nghĩa class list mà checkpoint có thể dự đoán? 
Class list do taxonomy và dữ liệu huấn luyện của checkpoint định nghĩa, ở đây là `ImageNet-1K`; không phải annotator tự tạo thêm lớp tại thời điểm suy luận. `class_id` 468 được ánh xạ sang tên `cab` theo metadata của checkpoint.
- Vì sao cần giữ cả ID, tên lớp và tên taxonomy? 
`class_id` là mã ổn định để máy đối chiếu; `class_name` giúp con người đọc và kiểm tra; `taxonomy_name` cho biết hệ quy chiếu/class list đang được dùng. Giữ đủ ba trường giúp tái lập, tránh nhầm tên lớp giữa các taxonomy hoặc giữa các phiên bản model.
- Nếu ảnh có nhiều chủ thể, guideline cần quy định điều gì? 
Guideline phải nói rõ nhiệm vụ là gán một nhãn cho toàn ảnh hay cho chủ thể chính, cách chọn nhãn khi có nhiều chủ thể, và trường hợp ảnh có nhiều lớp ngang nhau hoặc không có lớp phù hợp. Không được suy ra số lượng/vị trí từng chủ thể từ prediction cấp ảnh; ca mơ hồ cần đánh dấu để reviewer quyết định hoặc escalation.
- Vì sao model score không phải ground truth? 
Score 0.510915 là mức độ tin cậy tương đối do mô hình tính cho lớp `cab` trong các lớp mà checkpoint biết. Nó không phải nhãn đúng do người kiểm duyệt xác nhận, không chứng minh ảnh thực sự là taxi, và không thay thế ground truth được tạo theo guideline. Annotator/reviewer vẫn phải kiểm tra ảnh và ghi nhận đúng/sai hoặc ca không chắc chắn.

## 2. Phát hiện vật thể – lớp và box cho từng object

Nguồn evidence: `detection_predictions.json` và `visuals/detection_predictions.png`, sample `kitchen`.

- Một record (`class_name`, `score`, `bbox_xyxy`, `bbox_width`, `bbox_height`): 
`class_name`: `person`, 
`score`: `0.912625`, 
`bbox_xyxy`: `[385.33, 69.24, 498.92, 348.92]`, 
`bbox_width`: `113.58`, 
`bbox_height`: `279.68`.
- Diễn giải vị trí box bằng lời: 
Box bao quanh một người ở vùng bên phải ảnh `kitchen`, góc trên bên trái nằm ở tọa độ từ x = 385.33 đến x = 498.92 và góc dưới bên phải nằm ở tọa độ từ y = 69.24 đến y = 348.92. Box rộng khoảng 113.58 pixel và cao khoảng 279.68 pixel.
- So sánh số prediction ở hai threshold: 
Với các record `kitchen` trong file, có 11 prediction ở threshold gốc `0.35`. Nếu lọc hậu kỳ cùng danh sách theo threshold `0.70`, còn 3 prediction: 1 `person` và 2 `bowl`. Đây là so sánh độ lọc trên output, không phải chạy lại model và không phải ground truth. Nếu đặt threshold cao hơn, số lượng bounding box dự đoán sẽ giảm đi vì mô hình chỉ trả về các kết quả có độ chắc chắn rất cao. Ngược lại, nếu threshold thấp hơn, số lượng box sẽ tăng lên bao gồm cả các dự đoán kém tin cậy.
- Điều gì thay đổi đối với độ bao phủ và khối lượng reviewer cần xem? 
Threshold `0.35` giữ độ bao phủ cao hơn, gồm cả vật thể có score thấp như `bowl` `0.381215` và `cup` `0.381542`, nhưng reviewer phải xem 11 box và xử lý nhiều false positive hơn. Threshold `0.70` giảm số box cần xem xuống 3, nhưng có thể bỏ sót object nhỏ, bị che khuất hoặc khó nhận diện; vì vậy độ bao phủ giảm.
- Đề xuất một quy tắc box chặt: 
Box phải bao quanh toàn bộ phần nhìn thấy của đúng một object, sát biên ngoài của object, không bao gồm nền hoặc object bên cạnh; tọa độ phải nằm trong kích thước ảnh và dùng thống nhất định dạng `xyxy`. Không dùng score hoặc threshold để thay thế quy tắc vẽ box ground truth.
- Với object bị che khuất/cắt mép, điều gì cần guideline hoặc escalation quyết định?
Guideline cần định rõ mức độ hiển thị tối thiểu (ví dụ: chỉ vẽ box nếu thấy trên 30% vật thể), và cách vẽ box đối với phần bị che: chỉ vẽ phần nhìn thấy thực tế hay ước lượng kích thước cho toàn bộ vật thể bị che khuất một phần; và cách xử lý object bị cắt ở biên ảnh. Trường hợp không xác định được class, ranh giới hoặc object có bị trùng với object khác thì annotator đánh dấu mơ hồ; reviewer/QC quyết định hoặc escalation thay vì tự đoán.

## 3. Phân đoạn theo từng đối tượng – polygon cho mỗi instance

Nguồn evidence: `segmentation_predictions.json` và `visuals/segmentation_prediction.png`, sample `kitchen`.

- Một record (`instance_id`, `class_name`, `score`, số điểm và một phần `polygon_xy`): 
`instance_id`: `kitchen-001`, 
`class_name`: `person`, 
`score`: `0.899318`, 
`polygon_point_count`: `348`; 
một phần polygon bắt đầu bằng `[[446.0, 70.0], [445.0, 71.0], [444.0, 71.0], [443.0, 72.0], [442.0, 72.0]]`.
- Polygon bổ sung chi tiết gì so với box? 
Thay vì đóng khung hình chữ nhật, polygon cung cấp một chuỗi mảng các tọa độ (polygon_xy) mô tả chính xác ranh giới hình học và đường viền thực tế của vật thể với số điểm linh hoạt.
- `instance_id` dùng để làm gì và không phải loại ID nào? 
`instance_id` dùng để phân biệt từng object cụ thể trong một ảnh, liên kết polygon với class, score và các metadata của cùng prediction. Nó không phải `class_id`, không phải mã taxonomy, không phải ground-truth ID và không phải track ID theo dõi object qua nhiều frame.
- Đề xuất một quy tắc biên mask: 
Mask phải bao phủ liên tục phần nhìn thấy của đúng instance và bám sát biên quan sát được, không lấy nền, vật thể khác hoặc vùng bị che làm phần mask. Các điểm polygon phải theo cùng đơn vị pixel, nằm trong ảnh và được sắp xếp thành một đường biên hợp lệ.
- Với vùng mờ/tiếp xúc/che khuất, điều gì cần guideline hoặc escalation quyết định?
Với vùng mờ/tiếp xúc/che khuất, guideline cần quy định có vẽ phần bị che hay chỉ phần nhìn thấy, cách xử lý khe nhỏ và biên không rõ. Annotator đánh dấu vùng không chắc chắn; reviewer/QC kiểm tra tính nhất quán giữa các instance và escalation khi không thể xác định biên hoặc phần thuộc về object nào.

## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ | Đơn vị/định dạng ground truth | Lỗi hoặc điểm mơ hồ quan sát được | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một nhãn cho toàn ảnh, gồm `class_id`/`class_name` theo taxonomy và trạng thái chắc chắn nếu guideline yêu cầu | Prediction `cab` có score 0.510915 nhưng score không chứng minh nhãn đúng; ảnh có thể có nhiều chủ thể | Đọc guideline, chọn nhãn phù hợp cho toàn ảnh hoặc đánh dấu mơ hồ; không dùng score làm ground truth | Đối chiếu ảnh với guideline, kiểm tra taxonomy và các ca nhiều chủ thể/không có nhãn phù hợp |
| Phát hiện vật thể | Mỗi object là một class và box `xyxy` theo pixel | Threshold `0.35` giữ 11 prediction còn `0.70` giữ 3; box sát biên, object nhỏ, bị che hoặc cắt mép dễ gây khác biệt | Vẽ box chặt cho phần nhìn thấy theo rule, gán class và đánh dấu ca không chắc chắn | Kiểm tra class, số object, box có chứa nền/thiếu object không và các trường hợp cần escalation |
| Instance segmentation | Mỗi instance là một class và polygon pixel hợp lệ | Polygon cần bám biên; vùng tiếp xúc, mờ, che khuất hoặc sát mép ảnh có thể không xác định | Vẽ mask cho đúng instance, giữ `instance_id` để đối chiếu và đánh dấu biên mơ hồ | Kiểm tra polygon không lẫn nền/instance khác, tính hợp lệ của biên và tính nhất quán giữa các annotator |

## 5. An toàn dữ liệu

- Một quy tắc bảo vệ dữ liệu: Không tải xuống, chụp ảnh màn hình, hoặc chia sẻ bất kỳ hình ảnh và dữ liệu nhãn nội bộ nào ra bên ngoài hệ thống làm việc và dự án.
- Nếu thấy ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng và báo cho: giảng viên hoặc người phụ trách dữ liệu qua kênh nội bộ được chỉ định, đồng thời không tải xuống, sao chép hoặc chia sẻ thêm dữ liệu đó.

## 6. Danh sách bằng chứng

- [ ] `classification_predictions.json`
- [ ] `detection_predictions.json`
- [ ] `segmentation_predictions.json`
- [ ] `IMAGE_ATTRIBUTION.md`
- [ ] `visuals/classification_top5.png`
- [ ] `visuals/detection_predictions.png`
- [ ] `visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`.
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output.
