<<<<<<< HEAD
--- ui/events/devices/x11/device_data_manager_x11.cc.orig	2021-01-18 21:29:48 UTC
+++ ui/events/devices/x11/device_data_manager_x11.cc
@@ -840,6 +840,9 @@ void DeviceDataManagerX11::SetDisabledKeyboardAllowedK
=======
--- ui/events/devices/x11/device_data_manager_x11.cc.orig	2021-03-12 23:57:48 UTC
+++ ui/events/devices/x11/device_data_manager_x11.cc
@@ -841,6 +841,9 @@ void DeviceDataManagerX11::SetDisabledKeyboardAllowedK
>>>>>>> upstream/main
 }
 
 void DeviceDataManagerX11::DisableDevice(x11::Input::DeviceId deviceid) {
+#if defined(OS_BSD)
+  NOTIMPLEMENTED();
+#else
   blocked_devices_.set(static_cast<uint32_t>(deviceid), true);
   // TODO(rsadam@): Support blocking touchscreen devices.
   std::vector<InputDevice> keyboards = GetKeyboardDevices();
<<<<<<< HEAD
@@ -849,6 +852,7 @@ void DeviceDataManagerX11::DisableDevice(x11::Input::D
=======
@@ -850,6 +853,7 @@ void DeviceDataManagerX11::DisableDevice(x11::Input::D
>>>>>>> upstream/main
     keyboards.erase(it);
     DeviceDataManager::OnKeyboardDevicesUpdated(keyboards);
   }
+#endif
 }
 
 void DeviceDataManagerX11::EnableDevice(x11::Input::DeviceId deviceid) {
