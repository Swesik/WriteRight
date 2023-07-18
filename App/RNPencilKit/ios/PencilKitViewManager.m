//
//  PencilKitViewManager.m
//  RNPencilKit
//
//  Created by Rupesh Chaudhari on 22/04/23.
//
#import <React/RCTViewManager.h>
#import <React/RCTUIManager.h>
#import <PencilKit/PencilKit.h>
// Added for Storing the image in Photos
#import <Photos/Photos.h>

@interface PencilKitViewManager : RCTViewManager
@property PKCanvasView* canvasView;
// Added these two below to store the drawing and create an Image
@property PKDrawing* drawing;
@property UIImage* drawingImage;
@end

@implementation PencilKitViewManager

RCT_EXPORT_MODULE(PencilKit)

- (UIView *)view
{
  _canvasView = [[PKCanvasView alloc] init];
  _canvasView.drawing = _drawing; // Added this to store the user's drawing
  _canvasView.drawingPolicy = PKCanvasViewDrawingPolicyAnyInput;
  _canvasView.overrideUserInterfaceStyle = UIUserInterfaceStyleLight;
  _canvasView.multipleTouchEnabled = true;
  return _canvasView;
}

// Added below methods
RCT_EXPORT_METHOD(clearDrawing: (nonnull NSNumber *)viewTag)
{
  NSLog(@"Writing image to png");
  NSString *documentsDirectory = [NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES) lastObject];
  NSString *imageSubdirectory = [documentsDirectory stringByAppendingPathComponent:@"WrightRightImg"];
  NSString *filePath = [imageSubdirectory stringByAppendingPathComponent:@"MyImageName.png"];
  [UIImageJPEGRepresentation(self->_drawingImage, 1.0) writeToFile:filePath atomically:YES];
  NSLog(@"Clearing Drawing");
  [self clearDrawing];
}

-(void) clearDrawing{
  _canvasView.drawing = [[PKDrawing alloc] init];
}

RCT_EXPORT_METHOD(captureDrawing: (nonnull NSNumber *)viewTag)
{
  NSLog(@"Capturing Drawn Image");
  [self captureDrawing];
}

-(void) captureDrawing{
  dispatch_async(dispatch_get_main_queue(), ^{
    self->_drawingImage = [self->_canvasView.drawing imageFromRect:self->_canvasView.bounds scale:1.0];
    [[PHPhotoLibrary sharedPhotoLibrary] performChanges:^{[
      PHAssetChangeRequest creationRequestForAssetFromImage:self->_drawingImage];
    } completionHandler:^(BOOL success, NSError *error) {
      if (success) {
        NSString *path = [NSString stringWithFormat:@"photos-redirect://"];
        NSURL *imagePathUrl = [NSURL URLWithString:path];
        [[UIApplication sharedApplication] openURL:imagePathUrl options:@{} completionHandler:nil];
        [self clearDrawing];
      } else {
        NSLog(@"Error creating asset: %@", error);
      }
    }];
  });
}

@end
