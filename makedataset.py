import tarfile
import os

# Ayarları yapalım
archive_path = "D:/public_images.tar.bz2" # Dosyanın D'deki tam yolu
output_dir = "D:/images_sample/"
max_images = 5000 # İlk etapta test için 5 bin tane yeterli

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("Arşiv okunuyor, lütfen bekleyin (Bu işlem biraz zaman alabilir)...")

with tarfile.open(archive_path, "r:bz2") as tar:
    count = 0
    for member in tar:
        # Sadece görsel dosyalarını seçelim
        if member.isfile() and member.name.lower().endswith(('.jpg', '.jpeg', '.png')):
            # Klasör yapısını bozup sadece dosya adıyla kaydedelim
            member.name = os.path.basename(member.name) 
            tar.extract(member, output_dir)
            count += 1
            
            if count % 500 == 0:
                print(f"{count} görsel çıkarıldı...")
            
            if count >= max_images:
                break

print(f"İşlem tamam! {count} adet görsel {output_dir} klasörüne başarıyla çıkarıldı.")