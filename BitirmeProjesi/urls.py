from django.urls import path
from BitirmeProjesi import settings
from django.conf.urls.static import static
from BitirmeProjesi import views

urlpatterns = [
    path('', views.mainscreen, name='main_screen'),
    path('detail/<str:urun_adı>/', views.csv_detail, name='detail'),
    path('search/', views.search, name='search'),
    path('notification/', views.notification, name='notification'),
    path('predict/', views.predict, name='predict'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
