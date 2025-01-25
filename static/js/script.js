document.addEventListener("DOMContentLoaded", function () {
    const uploadOption = document.getElementById('upload-option');
    const cameraOption = document.getElementById('camera-option');
    const uploadSection = document.getElementById('upload-section');
    const cameraSection = document.getElementById('camera-section');
    const startCameraButton = document.getElementById('start-camera');
    const video = document.getElementById('video');
    const captureButton = document.getElementById('capture');
    const cameraSelect = document.getElementById('camera-select');
    const canvas = document.getElementById('canvas');
    const imageDataInput = document.getElementById('image_data');
    const cameraForm = document.getElementById('camera-form');
    let currentStream = null;

    // 📌 إظهار خيارات التحليل عند النقر على الأزرار
    uploadOption.addEventListener('click', function () {
        uploadSection.style.display = 'block';
        cameraSection.style.display = 'none';
    });

    cameraOption.addEventListener('click', function () {
        cameraSection.style.display = 'block';
        uploadSection.style.display = 'none';
        getCameras(); // جلب قائمة الكاميرات المتاحة
    });

    // 🚀 جلب قائمة الكاميرات المتاحة
    function getCameras() {
        navigator.mediaDevices.enumerateDevices()
            .then(devices => {
                cameraSelect.innerHTML = ''; // مسح القائمة السابقة
                devices.forEach(device => {
                    if (device.kind === 'videoinput') {
                        const option = document.createElement('option');
                        option.value = device.deviceId;
                        option.text = device.label || `Camera ${cameraSelect.length + 1}`;
                        cameraSelect.appendChild(option);
                    }
                });

                // التحقق إذا كانت هناك كاميرات متاحة
                if (cameraSelect.options.length > 0) {
                    startCamera(cameraSelect.value);
                } else {
                    alert("لا توجد كاميرات متاحة. يرجى التأكد من توصيل كاميرا.");
                }
            })
            .catch(err => {
                console.error("Error getting cameras:", err);
                alert("حدث خطأ في جلب الكاميرات. يرجى المحاولة مرة أخرى.");
            });
    }

    // 🎥 تشغيل الكاميرا المختارة
    function startCamera(deviceId) {
        if (currentStream) {
            currentStream.getTracks().forEach(track => track.stop()); // إيقاف البث السابق
        }

        const constraints = {
            video: deviceId ? { deviceId: { exact: deviceId } } : true
        };

        navigator.mediaDevices.getUserMedia(constraints)
            .then(stream => {
                currentStream = stream;
                video.srcObject = stream;
                video.style.display = 'block';
                captureButton.style.display = 'block';
            })
            .catch(err => {
                console.error("Error accessing camera:", err);
                alert("حدث خطأ في الوصول إلى الكاميرا. يرجى التأكد من صلاحيات الوصول.");
            });
    }

    // 📸 التقاط الصورة وتصغير حجمها قبل الإرسال
    captureButton.addEventListener('click', function () {
        if (!video.srcObject) {
            alert("لم يتم تفعيل الكاميرا.");
            return;
        }

        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        // ✅ تحويل الصورة إلى JPEG وضغطها إلى 85% جودة
        canvas.toBlob(function (blob) {
            const reader = new FileReader();
            reader.readAsDataURL(blob);
            reader.onloadend = function () {
                imageDataInput.value = reader.result;
                cameraForm.submit();
            };
        }, 'image/jpeg', 0.85);
    });

    // 🔄 تغيير الكاميرا عند اختيار كاميرا مختلفة
    cameraSelect.addEventListener('change', function () {
        startCamera(cameraSelect.value);
    });

    // إضافات لتحسين تجربة المستخدم
    // إظهار تحميل الصورة أثناء استخدام الكاميرا
    video.addEventListener('play', function () {
        captureButton.style.display = 'inline-block'; // إظهار زر التقاط الصورة عند تشغيل الفيديو
    });

    // تفعيل أو تعطيل زر التقاط الصورة حسب حالة الفيديو
    function toggleCaptureButton() {
        if (video.paused || video.ended) {
            captureButton.disabled = true;
        } else {
            captureButton.disabled = false;
        }
    }

    video.addEventListener('pause', toggleCaptureButton);
    video.addEventListener('ended', toggleCaptureButton);
});
