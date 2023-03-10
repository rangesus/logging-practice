//
//  AuthController.h
//  Coacheller-iOS
//
//  Created by John Smith on 4/9/13.
//  Copyright (c) 2013 Fanster. All rights reserved.
//

#import <Foundation/Foundation.h>
#import "FacebookAuthController.h"

@interface AuthController : NSObject <FacebookDataProtocol>
@property (strong, nonatomic) FacebookAuthController* facebookAuthController;
- (BOOL) isLoggedIn;

- (NSString*) getUserEmailAddress;
- (NSString*) getUserFirstName;
- (NSString*) getUserLastName;
- (NSString*) getFacebookID;

@end
