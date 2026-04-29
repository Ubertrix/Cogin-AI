# Mathematics Knowledge Database for the Mathematics Expert
# Contains mathematical rules and concepts in both Arabic and English

MATH_DATA = {
    "algebra": {
        "ar": [
            "حل المعادلات الخطية: ax + b = c",
            "قاعدة المربع الكامل: (a + b)^2 = a^2 + 2ab + b^2",
            "المتغيرات والثوابت في التعبيرات الجبرية"
        ],
        "en": [
            "Solving linear equations: ax + b = c",
            "Perfect square rule: (a + b)^2 = a^2 + 2ab + b^2",
            "Variables and constants in algebraic expressions"
        ]
    },
    "calculus": {
        "ar": [
            "قواعد الاشتقاق الأساسية: d/dx(x^n) = nx^(n-1)",
            "التكامل هو العملية العكسية للتفاضل",
            "حساب النهايات والاستمرارية للدوال"
        ],
        "en": [
            "Basic differentiation rules: d/dx(x^n) = nx^(n-1)",
            "Integration is the inverse process of differentiation",
            "Calculating limits and continuity of functions"
        ]
    },
    "geometry": {
        "ar": [
            "نظرية فيثاغورس: a^2 + b^2 = c^2 للمثلث قائم الزاوية",
            "مساحة الدائرة: πr^2",
            "حساب أحجام المجسمات الهندسية"
        ],
        "en": [
            "Pythagorean theorem: a^2 + b^2 = c^2 for right-angled triangles",
            "Area of a circle: πr^2",
            "Calculating volumes of geometric solids"
        ]
    }
}

# Additional training sentences to increase dictionary awareness of mathematics
MATH_CORPUS = [
    # English
    "Calculate the square root of 144.",
    "Solve the quadratic equation x^2 + 5x + 6 = 0.",
    "The derivative of sin(x) is cos(x).",
    "Geometry studies shapes, sizes, and properties of space.",
    "Arithmetic operations include addition, subtraction, multiplication, and division.",
    
    # Arabic
    "احسب الجذر التربيعي للرقم 144.",
    "حل المعادلة التربيعية س^2 + 5س + 6 = 0.",
    "مشتقة جيب الزاوية (جا) هي جيب التمام (جتا).",
    "الهندسة تدرس الأشكال والأحجام وخصائص الفضاء.",
    "العمليات الحسابية تشمل الجمع والطرح والضرب والقسمة."
]
